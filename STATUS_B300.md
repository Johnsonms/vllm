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
