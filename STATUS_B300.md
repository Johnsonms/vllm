# STATUS — B300 node (role `b300`)

## B1 — ack 2026-11-24
- Read README, setup/B300_START.md, TASKS_B300.md. Starting B1.
- Node: 8x NVIDIA B300 SXM6 AC, driver 595.112.07, CUDA toolkit 13.2. All 8 GPUs idle (0 MiB, 0%) at start.
- Background host load ~10 (system daemons, ~9 of 256 cores); no user GPU jobs.
- Next: build env at e66bede56b, then sanity tests, coverage read, decode profile.

## B1 — progress 2026-11-24
- Env built at e66bede56b (precompiled wheel), FlashInfer cubins + jit-cache (incl. sm103a) downloaded.
- Versions: torch 2.14.0+cu132, FlashInfer 0.7.4, Triton 3.8.0, cuBLAS 13.4.0.1, NCCL 2.30.7, vLLM 0.34.1rc1.dev61+ge66bede56; capability (10, 3) confirmed.
- Model config/tokenizer for Qwen3.6-35B-A3B-FP8 downloaded by name (no token needed; dummy weights).
- Push access: OK.
- Running: `tests/kernels/test_bf16_skinny_gemm.py` (789 items, CuTe compile-bound, ~30 min). Decode profile follows after it, on an idle node.

## B1 — progress 2026-11-24 (2)
- Sanity: `pytest tests/kernels/test_bf16_skinny_gemm.py -q` → **351 passed, 438 skipped** (4m55s, GPU 0).
- Coverage (read only): Qwen4Exp SM103 table = 12 shapes / 34 entries, TP4 (6 shapes) + TP1 (6 shapes), no TP2/TP8 (SM100 has 24 shapes). Generic `SKINNY_GEMM_PLANS_BY_CAPABILITY` has only `(12, 1)`: no `(10, 3)`. `FI_ALLREDUCE_FUSION_MAX_SIZE_MB[103]` = {2: 64, 4: 64, 8: 4, 16: 64}.
- TP8 AR backend on this node: `Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl` (no trtllm fallback), Qwen3-235B-A22B dummy TP8, `allreduce_rms` fusion enabled.
  - Note: Qwen3.6-35B-A3B-FP8 cannot load at TP8 ("input_size_per_partition = 64 is not divisible by weight quantization block_k = 128"); expected for this checkpoint, so the check used the 235B config.
- Running: TP1 decode profile (c1, c8). Node rule from the owner: if anyone else uses the GPUs, this node stops its jobs and steps back.

## B1 — DONE 2026-11-24
Full report: `results/b300/B1/SUMMARY.md`.
- Env OK at e66bede56b, capability (10, 3). Skinny GEMM test: 351 passed / 438 skipped.
- SM103 coverage: Qwen4Exp SM103 = TP1 + TP4 only (12 shapes / 34 entries); generic skinny table has no (10, 3); AR fusion cap TP8 = 4 MB; TP8 FlashInfer AR resolves to **mnnvl**.
- Qwen3.6-35B-A3B-FP8 TP1 decode, median TPOT: **c1 3.00 ms, c8 4.92 ms** (2 runs each, within 0.1%).
- **cuBLAS BF16 GEMMs = 18.0% of kernel time at c1, 12.4% at c8** (B200: 15.9 / 13.3%). 221 launches/step: router, GDN `in_proj_ba`, shared-expert gate, LM head + 70-110 splitKreduce. Without the aux-stream shared-expert gate: 14.6% / 10.3%.
- Waiting for the manager's review before B2. Polling every ~10 min.
