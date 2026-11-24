# W12 — fewer `per_token_group_quant` launches in Qwen3.6-35B-A3B-FP8 decode on B200

Node: 8x B200, vLLM main @ 8399da3780 (precompiled), TP1, dummy weights. No vLLM file edited.
Status: **steps 1-2 posted (19:25Z). No A/B run: no flag or gating-only change enables a fusion.** Waiting for the manager's decision.

## Result
- All 200 launches are mapped (section 1). Each layer has 5: attention input, attention output projection, shared-expert `gate_up_proj`, shared-expert `down_proj`, routed-experts input.
- **None of the existing fusions can fire for this checkpoint on NVIDIA, for two separate reasons:**
  1. The compile passes (`fuse_norm_quant`, `fuse_act_quant`) never run. `Qwen3_5Moe*` is a default breakable-CUDA-graph architecture, which forces `CompilationMode.NONE` (#51012). The startup line "Enabled custom fusions: norm_quant, act_quant" is printed anyway.
  2. The manual (non-compile) fusions need the consumer linear to accept a pre-quantized input. The block-FP8 linears here use `CutlassFp8BlockScaledMMKernel`, which does not override `input_quant_key()` (base returns `None`), so every producer falls back to the plain activation.
- **Smallest real change:** let the block kernel accept `kFp8Dynamic128Sym` input. The shared-expert `down_proj` launch (40 per step) then fuses with code that is already in main: `Qwen3MoeMLP.forward` already calls `maybe_fused_act_quant`, and `(SiluAndMul, kFp8Dynamic128Sym)` is already registered on `silu_and_mul_per_block_quant`. This is a kernel-interface change, not a flag, so I did not start the A/B.
- The other 160 launches each need a producer kernel change as well (section 2).

## 1. Launch -> producer map

Method: wrapped `torch.ops._C.per_token_group_fp8_quant` in-process and recorded the Python stack, the owning module, the input pointer and the arguments during CUDA-graph capture at 1 token. 200 calls per capture, group size 128 everywhere, column-major scales everywhere.

| # in layer | consumer (caller of the quant) | launches / step | K | producer of the input | UE8M0 scales |
|---|---|---|---|---|---|
| 1 | `linear_attn.in_proj_qkvz` (30) or `self_attn.qkv_proj` (10) | 40 | 2048 | input RMSNorm (`Qwen3_5RMSNorm.forward_fused`, FlashInfer fused add + RMSNorm) | no |
| 2 | `linear_attn.out_proj` (30) | 30 | 4096 | GDN gated RMSNorm (`_fused_gdn_gated_rms_norm_kernel`) | no |
| 2 | `self_attn.o_proj` (10) | 10 | 4096 | attention output gate (`_fused_sigmoid_gate_mul_kernel`) | no |
| 3 | `mlp.shared_expert.gate_up_proj` | 40 | 2048 | post-attention RMSNorm | no |
| 4 | `mlp.shared_expert.down_proj` | 40 | 512 | `SiluAndMul` (`act_and_mul_kernel`) | no |
| 5 | routed experts (`fused_moe/utils.py:_fp8_quantize` via `MoERunner`) | 40 | 2048 | post-attention RMSNorm, **same tensor as #3** (same pointer) | **yes** |

- Callers 1-4 go through `QuantFP8.forward_cuda` from `BlockScaledMMLinearKernel.apply_weights` (`self.apply_input_quant`).
- Launches 3 and 5 quantize the same activation with the same group size and layout. They differ only in `use_ue8m0`: the routed-experts call passes `None`, which resolves to `is_deep_gemm_e8m0_used()` (true on this node), although the routed kernel is the TRT-LLM one. So the two outputs are not the same bytes and cannot be shared as is. Whether power-of-two scales are intended for the TRT-LLM MoE input I did not check.

## 2. Fusion status per launch

| launch | fused kernel in main? | wired for this model? | why it does not fire |
|---|---|---|---|
| 1 (norm -> qkvz / qkv) | Manual: `gemma_add_rms_norm_quant` (Triton, model-local) supports static-tensor FP8, per-token FP8 and NVFP4 only. Compile: `rms_norm_per_block_quant` + `RMSNormGroupQuantPattern`. | Manual path is wired (`forward_fused` checks `_norm_quant_key(linear)`). | `_norm_quant_key` gets `None`: the block kernel exposes no key, and group-128 is not one of the three modes. The compile pattern never runs (mode NONE) and skips fp32 Gemma norm weights anyway (#65497, open, fixes that for compiled models only). |
| 2 (GDN gated norm -> out_proj) | Manual: `_fused_gdn_gated_rms_norm_quant` quantizes only for `kFp8StaticTensorSym`. | Yes | Key is not static-tensor (and is `None`). |
| 2 (attention gate -> o_proj) | None found. | No | No producer. |
| 3 (norm -> shared gate_up) | Same kernels as launch 1. | Not wired: the post-attention norm output also feeds the router, the gate and the routed experts. | No call site, and the key is `None`. |
| 4 (SiluAndMul -> shared down) | **Yes: `silu_and_mul_per_block_quant` (C++), registered for `(SiluAndMul, kFp8Dynamic128Sym)` in `layers/fusion/fused_act_quant.py`.** | **Yes: `Qwen3MoeMLP.forward` calls `maybe_fused_act_quant(act_fn, gate_up, down_proj)`.** | Only the key: `get_input_quant_key(down_proj)` is `None`. |
| 5 (norm -> routed experts) | None for the TRT-LLM MoE input. | No | No producer; UE8M0 mismatch with launch 3. |

What launch 4 would need (described, not applied):
- `CutlassFp8BlockScaledMMKernel.input_quant_key()` returns `kFp8Dynamic128Sym` when the activation key is that and no token padding is needed.
- `BlockScaledMMLinearKernel.apply_weights` takes the `QuantizedActivation` instead of calling `self.quant_fp8`.
- Scale layout: the kernel's own quant writes column-major scales; the registered producer calls the op with `is_scale_transposed=False`. The op has the flag, so the producer or the kernel must agree on one layout. At 1 token they coincide.
- Expected gain from the profile: 40 launches of about 2.4 us (profiled) out of a 3,390 us step, so **about 2-3 % at 1 request** for launch 4 alone, if the fused kernel is not slower than `act_and_mul_kernel`. Not measured.

## 3. Duplicates (open upstream PRs)
- **#61073** "Extend QuantizedActivation contract with get_input_quant_scale and maybe_fused_norm_quant": adds a manual norm + quant fusion that lists `kFp8Dynamic128Sym` among its keys, wired in `llama.py`. Closest to launches 1 and 3. It does not touch the block kernel's `input_quant_key` in its file list, and not Qwen3.5.
- **#65497** "Fuse RMSNorm + block FP8 quant for fp32 (Gemma-style) norm weights": compile path only; its text says Qwen3.5 on NVIDIA is not affected.
- **#45770** "Support mixed input/weight dtype in fused RMSNorm+FP8 quant kernels (fixes Qwen3.5-FP8)": older, compile path, last updated in August.
- #43944 (ROCm manual RMS + group quant for Qwen3) and #68151 (RDNA4 SiLU-mul + block FP8) are the ROCm counterparts.
- Found no open PR that gives the CUDA block-FP8 kernel a pre-quantized input.

## 4. Side observation from the same trace
`Qwen3MoeMLP.forward` computes `F.sigmoid(self.expert_gate(x)[0]) * out`: this is the gate GEMM + torch sigmoid + torch multiply that the W11 profile showed 40 times each. W11a covers the GEMM; the sigmoid and the multiply stay two launches per layer.

## Options for the manager
1. Launch 4 only (kernel input key + `apply_weights`): smallest, about 30 lines in two shared files, no new kernel. I can prototype it locally and run the serving A/B.
2. Launch 4 plus launch 1 via a group-128 mode in the model-local Triton norm-quant kernel: larger, a new kernel mode.
3. Stop here.
