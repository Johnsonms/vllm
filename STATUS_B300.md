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

## B2 — ack 2026-11-24
- Read the B1 review. Starting B2: 14 W11 shapes, M ∈ {1, 2, 4, 8}, sweep then 4-seed × 3-round verify; also reporting whether each W11 SM100 config qualifies on SM103.
- Real-weight download of Qwen3.6-35B-A3B-FP8 started in the background (files by name).

## B2 — sweep done 2026-11-24
- Manager's `sweep_skinny_generic.py`, 8 processes (one per GPU: {small, LM-head} × M ∈ {1,2,4,8}), CUPTI, CUDA graph, cold L2. 56/56 cells, no errors. The weight download (host/network only) overlapped the sweep; the verify runs on a quiet node.
- **49 entries on 14 shapes pass ≥ 1.10× — exactly the same cells as W11 on SM100.** Dropped: every LM head at M=8, and (248320, 2048) / (124160, 2048) at M=4 (1.04× / 1.07×).
- Small shapes 1.56–3.28×; LM heads 1.13–1.44× (M=1–2), 1.13–1.23× at M=4 for ≤ 62080 rows.
- Only 8 of 49 winning configs equal the W11 SM100 config. The verify times cuBLAS vs SM103 proposal vs W11 SM100 config side by side (`verify_sm103.py`), 4 seeds × 3 rounds, GPUs 0-3.
- Raw: `results/b300/B2/sweep.jsonl`, `proposed.json`.

## B2 — DONE 2026-11-24
Full report: `results/b300/B2/SUMMARY.md` (+ `verified.json`, `verify_summary.json`, raw `verify.jsonl`).
- **All 49 SM103 entries verify** (4 seeds × 3 interleaved rounds, CUPTI, CUDA graph, cold L2, GPUs 0-3). Same 49 cells as W11.
  - Small projections 1.50–3.12×; LM heads 1.13–1.46×.
  - Error vs fp32 equals cuBLAS in all 196 records (max 3.8e-3).
- **Shared-table question:** all 49 W11 SM100 configs also pass ≥ 1.10× on SM103.
  - sm100_us / b300_us: median 1.000, range 0.973–1.049. The worst cells are LM heads at M=1 (4.9% and 3.4% slower).
  - Thinnest SM100 margin on SM103: (62080, 2048) M=4 at 1.11× (min 1.10).
  - Registering (10, 3) to the SM100 table is defensible; a separate table gains ≤ ~5% on a few LM-head cells.
- Real weights downloaded (35 GB). Ready for the E2E A/B when the manager's table is up. Polling every ~10 min.

## B3 — ack 2026-11-24
- Read the B2 review (shared table). Setting up arms: base = e66bede56b, pr = 81159c69d6 + local one-line (10, 3) registration (not committed anywhere). TP1 real weights, c = 1, 2, 4, 8, order base/pr/base/pr, GSM8K per arm.

## B3 — progress 2026-11-24
- TP1 set 1: a run-counter bug in my script (loop variable shared with the cleanup function) made round 4 (pr) overwrite round 2 (pr). Kept: base r1, base r3, pr r4. All three code checks are as expected, AOT = 0. GSM8K: 0.963 / 0.962 (base), 0.961 (pr).
- Fixed the script and rerunning a clean base/pr/base/pr set 2 now. Set 1 will be reported as supplementary.

## B3 — TP1 done 2026-11-24 (set 2, clean A/B/A/B)
- Median TPOT, mean of 2 runs per arm, base → pr: c1 3.000 → 2.928 (**−2.4%**), c2 3.339 → 3.231 (**−3.2%**), c4 4.182 → 3.703 (**−11.5%**), c8 4.762 → 4.584 (**−3.7%**).
- Per-run spread is ≤ 0.03 ms within each arm. Set 1 (3 surviving runs) agrees: −2.3 / −3.3 / −10.8 / −4.5%.
- GSM8K: base 0.964 / 0.958, pr 0.964 / 0.964. Code checks OK in every arm; AOT = 0.
- c4 stands out: base c2→c4 adds 0.84 ms vs 0.58 ms c4→c8. That suggests a slow cuBLAS choice at M=4 on SM103; not profiled yet.
- Running the optional TP2 A/B (c1, c8) now.

## B3 — DONE 2026-11-24
Full report: `results/b300/B3/SUMMARY.md` (+ `runs.json`).
- TP1 (clean A/B/A/B): TPOT **−2.4% c1, −3.2% c2, −11.5% c4, −3.7% c8**.
- TP2 (optional): **−3.8% c1, −5.1% c8**. The two c1 base runs differ by 3%.
- GSM8K unchanged in every arm (0.958–0.964, 0 invalid). Code-check lines and AOT = 0 confirmed in all 12 runs.
- The c4 gain at TP1 is unusually large (base is slow at batch 4 on SM103). Not profiled; I propose a c4 profile of both arms if the manager wants the cause.
- Idle; polling every ~10 min.

## Manager update (11-25) — ack 2026-11-25
- Read the 11-25 update. Node idle (8 GPUs, 0 MiB). Polling resumed (~10 min).
- Queue: **B4** (SM103 verify of dense-Qwen3 91 entries + Qwen3-MoE 49 entries, cuBLAS vs given config, 4 seeds × 3 rounds, one seed per GPU) → B5 (Qwen3-32B TP8 AR-fusion cap 4/8/16/32 MB) → B6 (c2/c4/c8 decode profile, base vs (10, 3) table).
- Starting B4 now.
