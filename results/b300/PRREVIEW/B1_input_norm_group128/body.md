**Depends on #69778; the diff includes its commits until it merges.** This PR's own change is the last commit (`e77ca1439c`, 4 files, +134/−12).

## Overview

On NVIDIA, Qwen3.5 / Qwen3.6 fuse the consumer linear's input quant into the decoder `input_layernorm` (`gemma_add_rms_norm_quant`). That fusion covers static FP8, dynamic per-token FP8 and NVFP4, but not dynamic per-token-group-128 FP8.

Block-FP8 checkpoints such as Qwen3.6-35B-A3B-FP8 quantize their activations that way. So on those checkpoints, every `input_layernorm` is followed by a standalone `per_token_group_fp8_quant` launch before GDN `in_proj_qkvz` or attention `qkv_proj`.

#69778 lets the CUTLASS block-FP8 linear accept a pre-quantized group-128 activation. This PR adds a group-128 epilogue to `gemma_add_rms_norm_quant` so the input norm produces that activation directly:
- **Scale layout.** One scale per 128 columns, written column-major (`(M, H // 128)` with strides `(1, M)`), the layout the CUTLASS kernel reads.
- **Numerics.** The numerics are those of `per_token_group_quant_fp8` without UE8M0, computed on the bf16-rounded normed activation: `scale = max(amax, 1e-10) / 448`, then `x * (1 / scale)`.
- **Wiring.** `_norm_quant_key` returns `kFp8Dynamic128Sym` when the consumer reports it and `input_size_per_partition % 128 == 0`.
- **TP > 1.** The norm runs inside the FlashInfer all-reduce kernel, so `per_token_group_quant_fp8` (column-major, `use_ue8m0=False`, as in the CUTLASS linear's own `QuantFP8`) follows it. This is the same shape as the existing per-token FP8 path.

Not affected:
- static / per-token FP8 and NVFP4 checkpoints;
- non-NVIDIA platforms;
- layers whose linear does not report `kFp8Dynamic128Sym` (for example when the CUTLASS kernel needs token padding).

The GDN gated norm (→ `out_proj`) and the attention sigmoid gate (→ `o_proj`) are not touched here; see Related PRs.

## Results

**End to end: Qwen3.6-35B-A3B-FP8, TP=1, real weights, 1× B300 (SM103).**

Method:
- Base: #69778 head (`152bca1e65`). PR: this branch (`e77ca1439c`). `vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --language-model-only` with `VLLM_DISABLE_COMPILE_CACHE=1`. "Directly load AOT" appears 0 times in all four server logs.
- Both servers run at the same time on two GPUs. Short `vllm bench serve` bursts alternate between them: random 512 in / 512 out, `--ignore-eos`, 5 × concurrency prompts, order flipped each round, 6 rounds per concurrency.
- Then both servers restart with the GPUs swapped and the bursts run again. The table averages the two passes.

| Concurrency | base → PR, pass 1 | base → PR, pass 2 (GPUs swapped) | Mean | Pairs faster |
|---:|---|---|---:|---:|
| 1 | 2.986 → 2.908 ms (−2.61%) | 2.985 → 2.911 (−2.51%) | **−2.56%** | 12 / 12 |
| 2 | 3.373 → 3.305 (−2.00%) | 3.367 → 3.305 (−1.83%) | **−1.92%** | 12 / 12 |
| 4 | 4.199 → 4.167 (−0.76%) | 4.194 → 4.167 (−0.64%) | **−0.70%** | 12 / 12 |
| 8 | 4.713 → 4.632 (−1.72%) | 4.716 → 4.636 (−1.69%) | **−1.70%** | 12 / 12 |
| 16 | 5.669 → 5.629 (−0.71%) | 5.673 → 5.625 (−0.84%) | **−0.78%** | 12 / 12 |
| 32 | 6.942 → 6.856 (−1.24%) | 6.901 → 6.805 (−1.38%) | **−1.31%** | 12 / 12 |

Median TPOT.

**Kernel level.** The input norm at Qwen3.6-35B-A3B's hidden size of 2048, with a fused residual add:
- Unfused: FlashInfer `gemma_fused_add_rmsnorm` + `per_token_group_quant_fp8`. Fused: `gemma_add_rms_norm_quant`.
- 1× B300, CUDA graphs (20 calls per replay), L2 flushed before each replay, 7 interleaved rounds. The table shows the median over 3 GPUs; the range is across GPUs.

| M | unfused µs | fused µs | speedup |
|---:|---:|---:|---|
| 1 | 3.28 | 1.84 | 1.83× (1.78–1.87) |
| 2 | 4.30 | 1.85 | 2.33× (2.32–2.33) |
| 4 | 4.71 | 1.84 | 2.55× (2.50–2.55) |
| 8 | 4.61 | 2.05 | 2.25× (2.25–2.30) |
| 16 | 4.71 | 2.15 | 2.19× (2.19–2.19) |
| 32 | 4.72 | 2.15 | 2.19× (2.05–2.20) |
| 64 | 4.91 | 2.15 | 2.29× (2.28–2.29) |
| 128 | 4.92 | 2.08 | 2.36× (2.29–2.39) |
| 512 | 5.52 | 3.69 | 1.50× (1.47–1.50) |
| 2048 | 11.37 | 8.49 | 1.34× (1.34–1.34) |
| 8192 | 34.92 | 27.86 | 1.25× (1.25–1.25) |

No shape is slower.

**Accuracy.** The fused codes and scales equal `per_token_group_quant_fp8` applied to the same bf16 normed activation (test below). The model output is not bit-identical to the base, though. FlashInfer's norm and this Triton norm reduce in a different order, so at prefill sizes a few bf16 normed values round differently (for example 7 of 614k at M = 300, 128 of 16.8M at M = 8192).

- **Greedy, 256 tokens.** 3 / 32 single prompts and 19 / 256 batched prompts are identical to the base. The base against itself gives 32 / 32 and 176 / 256 (batched greedy is not deterministic run to run).
- **GSM8K.** `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64`, 1,319 questions.
  - 16 runs per arm, in four groups of 4, with the GPUs swapped between groups.
  - Accuracy: base 0.9618 (sd 0.0021), PR 0.9612 (sd 0.0016), Δ −0.05 points (Welch t −0.81).
  - Per question: 23.4 flips per base/PR run pair, against 17.1 (base/base) and 16.2 (PR/PR).
  - No question is right in all 16 base runs and wrong in all 16 PR runs. One question is the reverse.

## Testing

```
pytest -q tests/models/qwen3_5/test_gemma_add_rms_norm_quant.py \
  tests/kernels/core/test_fused_quant_layernorm.py \
  tests/kernels/core/test_fused_sigmoid_gate_mul.py \
  tests/kernels/mamba/test_gdn_output_projection.py
```
- Base (#69778 head) gives 819 passed; this PR gives 855 passed.
- The 36 new tests are `test_gemma_add_rms_norm_group128_quant_matches_chain`. They cover M ∈ {1, 7, 128, 129, 384, 385} × H ∈ {2048, 4096, 5120} × with and without residual, and use `torch.equal` on codes and scales. On the base code all 36 fail.
- `pytest -q tests/models/qwen3_5/test_fused_allreduce_norm_quant.py` (TP=2): 1 passed on both base and this PR.
- `pre-commit run --files <changed files>`: passes with no changes.

All on B300 (SM103), CUDA 13.2, torch 2.14.

Not measured: SM100 (B200) end to end of this branch, TP > 1 end to end, and other Qwen3.5 sizes.

## Related PRs

- **#69778** (this PR's base): makes the CUTLASS block-FP8 linear accept pre-quantized group-128 input, and fuses SiLU-and-mul → `down_proj`. This PR adds the input-norm producer for the same interface.
- **#69812**: fuses dynamic **per-token** FP8 into the GDN gated RMSNorm and the attention sigmoid gate. It does not cover group-128 and does not touch `gemma_add_rms_norm_quant`, so the two PRs are independent. A group-128 epilogue for those two producers is a natural follow-up once #69812 lands.
- **#65497, #36016, #45770**: Gemma RMSNorm + FP8 quant fusions in the torch.compile `rms_quant_fusion` pass and its C++ kernels. Qwen3.5 on NVIDIA runs with compile mode NONE (#51012), so those passes do not run there, and none of them changes `vllm/models/qwen3_5/`.

## AI assistance

Claude Code was used to write the kernel epilogue and tests and to run the benchmarks; the submitter reviewed every changed line.
