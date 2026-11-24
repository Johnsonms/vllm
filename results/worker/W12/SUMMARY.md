# W12 — fewer `per_token_group_quant` launches in Qwen3.6-35B-A3B-FP8 decode on B200

Node: 8x B200, vLLM main @ 8399da3780 (precompiled), TP1, dummy weights. No vLLM file edited.
Status: **steps 1-2 posted (19:12Z). No A/B run: no flag or gating-only change enables a fusion.** Waiting for the manager's decision.

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

---

# W12b — prototype of launch 4 (interim posted 20:22Z; final results in the last section, 21:04Z)

Local, env-gated edit on 8399da3780; not pushed anywhere; reverted when the measurements end. No code is included here, by choice: this branch is public.

## Result so far
- **It works and it is faster: TPOT p50 -5.0 % at 1 request, -2.3 % at 4** (real weights, TP1, 512/512, base/prototype alternating twice; run-to-run spread under 0.1 %). 8 requests is not settled: one base round was 5 % off; a second full A/B on an idle node is running.
- Quant launches per step 200 -> 160; the fused SiLU-mul + block-quant kernel runs 40 times instead of the activation kernel plus the quant kernel.
- **Greedy tokens are NOT equal to the base with real weights.** The fused kernel quantizes the fp32 product; the unfused path rounds it to BF16 first. This is a property of the existing fused kernel, not of the wiring.
- **The key change is not local to Qwen3.6.** The same linear kernel serves other quantization methods that already expose input keys, so their SiLU-mul -> down-projection paths would start fusing too (list below).
- **The registered producer has the wrong scale layout for this kernel.** Row-major scales give wrong GEMM output at 2 tokens and more; column-major is required.

## The edit, in words (5 files, 48 lines added)
1. The cutlass block-FP8 linear kernel reports the dynamic group-128 FP8 key as its accepted input key when its activation key is that and it pads no tokens, and declares that it wants column-major scales.
2. The shared block-scaled `apply_weights` accepts a pre-quantized activation and then skips its own input quant.
3. The `fp8` linear method exposes its kernel's input key on the layer after weight processing. **On main this method never exposes a key**, unlike the modelopt, online-FP8, compressed-tensors and Quark methods, so for `quant_method: fp8` checkpoints no manual fusion can fire with any kernel.
4. The expose helper also stores the kernel's scale layout on the layer.
5. The SiLU-mul block-quant producer allocates transposed scales and sets the op's transposed flag when the consumer asks for column-major.

## Scale layout (kernel ops called directly, N = 2048, K = 512)
GEMM output of the fused producer against the unfused path, max abs difference (outputs are about 0.3-0.5 in magnitude):

| tokens | scales column-major | scales row-major (what the registered producer writes today) |
|---|---|---|
| 1 | 0.0027 | 0.0027 |
| 2 | 0.0056 | **0.150** |
| 4 | 0.0039 | **0.225** |
| 8 | 0.0053 | **0.152** |
| 64 | 0.0082 | **0.310** |
| 2048 | 0.0117 | **0.796** |

So enabling the key without fixing the producer's layout would be silently wrong from 2 tokens up.

## Correctness
- **Layer level** (real row-parallel linear + `fp8` method + the fusion helper, synthetic block-FP8 weight, fp32 reference): the fusion fires at 1, 2, 4, 8, 64 and 2,048 tokens with column-major scales. Relative RMS error against fp32: unfused 2.15-2.41 %, fused 2.16-2.41 %, equal within 0.02 points at every size. Max relative error 2.0-2.7 % in both.
- **Not bit-identical:** 2.2-2.7 % of the quantized bytes differ and the group scales differ by up to about 1 % (same test as the layout table).
- **Greedy tokens, real weights** (4 chat prompts, 256 tokens, once one at a time and once as a batch of 4):
  - base run twice: identical in all 8 cases.
  - base vs prototype: different in all 8; first differing token at 48, 13, 20, 5 (single) and 0, 103, 20, 24 (batch).
  - base single vs base batch of 4, same prompts: first differing token at 48, 13, 91, 5. Three of the four positions are the same as base vs prototype, so those are near-ties that any numeric change flips; the base output already depends on batch composition.
  - Both arms give coherent text of the same kind. No quality benchmark was run.
- **Dummy weights:** tokens also differ, as expected with random logits; not informative.
- **Existing tests** (manual SiLU-mul quant fusion test, `tests/fusion`, the two block-FP8 test files, the cutlass scaled-mm test file): 813 passed, 407 failed, 675 skipped with the gate on **and** with it off. The failures I sampled are "Int8 not supported on SM100"; I compared the counts, not the full list of failing IDs. The manual-fusion and `tests/fusion` files pass. The helion test file cannot be imported here.

## Who else would start fusing
Call sites of the act-quant fusion helper whose activation is SiLU-and-mul fuse as soon as their down-projection gets this kernel with a dynamic group-128 activation:
- the Qwen3 MoE MLP class (dense Qwen3 MLP, and the shared expert of Qwen3-Next / Qwen3.5 / Qwen3.6),
- the Llama MLP, Nemotron-H, Pixtral (two sites), the transformers GLU fuser, the Muse Spark draft MLP.
- With edit 1 alone this already applies to checkpoints loaded through modelopt, online FP8 and compressed-tensors W8A8, which expose keys today. Edit 3 adds `quant_method: fp8` checkpoints, and for those it also exposes the static-tensor and per-token keys of the non-block cutlass and FlashInfer kernels for the first time.
- Not affected: the Qwen3.5 model-local norm fusions (they accept only static-tensor, per-token and NVFP4 keys). The cutlass kernel is third in the CUDA block-kernel list after the two DeepGEMM-based ones, so models that pick those are unchanged.
- All of them would get the same non-bit-identical numerics.

## Serving A/B, first run (real weights instead of dummy, since they are now on the node)
TP1 server, language model only, max length 4096; random 512/512, ignore-eos; order base, prototype, base, prototype on one GPU; compile cache disabled, AOT-load count 0 in all four logs. The test suite above was running on another GPU during this run.

| conc. | TPOT p50 base r1 / r2 (ms) | prototype r1 / r2 | change of the means |
|---|---|---|---|
| 1 | 3.540 / 3.542 | 3.363 / 3.363 | **-5.0 %** |
| 4 | 4.472 / 4.470 | 4.367 / 4.368 | **-2.3 %** |
| 8 | 5.183 / 4.923 | 4.977 / 4.988 | not settled (-1.4 % on means; +1.2 % against base r2 alone) |

The 1-request gain (178 us per token) is larger than my estimate from 40 quant launches (about 96 us): the fused kernel also replaces the separate activation kernel, and two output allocations per layer go away. I did not break this down further.


---

# W12b — final (21:04Z): answers to the four review points

The local edit is reverted; the vLLM tree is clean at 8399da3780; no worker process is left on any GPU. The diff is kept on the node only.

## 1. Idle-node A/B (real weights, TP1, 512/512, base / prototype alternating twice, nothing else running)

| conc. | TPOT p50 base r1 / r2 (ms) | prototype r1 / r2 | change of the means | output tok/s base -> prototype |
|---|---|---|---|---|
| 1 | 3.539 / 3.545 | 3.363 / 3.363 | **-5.05 %** | 277.7 -> 292.3 |
| 4 | 4.472 / 4.474 | 4.376 / 4.370 | **-2.24 %** | 875.8 -> 896.7 |
| 8 | 5.174 / 5.165 | 4.966 / 4.990 | **-3.70 %** | 1506.6 -> 1568.7 |

- AOT-load count 0 in all four logs. Spread between the two rounds of an arm: at most 0.5 %.
- 1 and 4 requests agree with the first (contended) run to 0.1 point. At 8 requests the first run's base r2 (4.923 ms) was the outlier; the idle run settles it.
- TTFT p50 unchanged within its noise (31-32 ms at 1 request, 51-54 at 4, 66-69 at 8).

## 2. Reconciling with the aux-stream finding
- **Order taken here:** `MULTI_STREAM_OVERLAPPED` at 1, 2, 4, 8 and 16 tokens (threshold 256), `NO_OVERLAP` at 512 and above. Same as the manager saw.
- **Diagnostic with `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`** (one round per arm, idle node): TPOT p50 base -> prototype 4.328 -> 4.240 ms at 1 request (**-2.0 %**), 5.384 -> 5.264 at 4 (-2.2 %), 5.988 -> 5.884 at 8 (-1.7 %). So the gain exists without the stream and is about 88 us per token at 1 request, i.e. 2.2 us for each of the 40 removed launches. (The stream itself is worth 18 % at 1 request: 4.328 vs 3.540 ms.)
- **Why the gain survives, and is larger, with the stream on: the shared expert is the longer branch at 1 token.** In the W11 decode trace of the unmodified code (1 request, 3 replays), in every one of the 40 layers the last shared-expert kernel on the aux stream (the gate multiply) ends after the last routed-experts kernel on the main stream (finalize): median 11.1 us later, range 8.4-13.5 us. The main stream waits for it. Anything removed from the shared branch therefore shortens the step until that 11 us lag is used up.
  - Caveat: the lag is measured under the profiler, which inflates this step from 3.39 to 4.64 ms.
  - I cannot explain from here why the manager's S (three launches less in the same branch) showed nothing. By this trace it should help too. Two things worth checking on the manager node: whether the same lag exists in a trace there, and whether the Triton replacement is slower per call than the three kernels it removes.
  - Not explained: why the saving is 4.4 us per layer with the stream but 2.2 us without. A guess, not tested: small kernels on two overlapping streams each run slower than alone.

## 3. Accuracy: GSM8K, real weights
`gsm8k_eval.py`, chat completions, `enable_thinking: false`, max tokens 1024, temperature 0, concurrency 64, 1,319 questions, 5 shots; order base, prototype, prototype, base.

| arm | run 1 | run 2 |
|---|---|---|
| base | 0.9598 | 0.9606 |
| prototype | 0.9613 | 0.9583 |

No invalid responses in any run. The four values span 4 questions; the two arms are not distinguishable. (Runs of one arm differ because the batch composition differs at concurrency 64.)

## 4. Is the row-major producer reachable on main today? **No.**
- On main at 8399da3780 no linear kernel returns the dynamic group-128 FP8 key from `input_quant_key()`. The keys returned are static-tensor FP8, per-token FP8, MXFP8 and NVFP4 only (checked every `input_quant_key` implementation under `kernels/linear`).
- `expose_input_quant_key` is the only place that sets the layer key, and it copies the kernel's answer.
- So the `(SiluAndMul, group-128)` registry entry has no consumer among linear layers, on any of modelopt, online FP8, compressed-tensors W8A8 block, Quark or the DeepGEMM-based kernels. **It is not a live bug; it is a trap for whoever adds the first consumer**, because the cutlass block kernel reads scales column-major and the entry writes them row-major. No reproduction exists on main; the layout table in the interim section is from calling the two ops directly.
- The compile-path pattern for the same op is a separate code path; I did not check which layout it passes.

## Summary for the decision
- Gain: -5.1 % / -2.2 % / -3.7 % TPOT p50 at 1 / 4 / 8 requests on Qwen3.6-35B-A3B-FP8 TP1, real weights, idle node.
- Cost: not bit-identical (greedy tokens differ from base; GSM8K unchanged); five shared files; the same fusion turns on for other SiLU-and-mul models that use this kernel; the producer's scale layout must be fixed in the same change.
- Not tested: TP > 1, the 397B model, prefill-heavy load, any model other than Qwen3.6-35B.
