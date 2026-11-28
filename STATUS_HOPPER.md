# Status: Hopper node (role `hopper`)

The Hopper node writes this file. Newest entries at the bottom.

## 2026-11-28 — H1 ack
- H1 ack, running. Node: 8x H100 80GB HBM3 (SM90), all pairs NV18 (full NVLink mesh), driver 595.91.07 (CUDA 13.2 driver). No system CUDA toolkit `bin/` (no nvcc) on this node; relying on pip-provided CUDA libs.
- Building env on base `cfbc42099a` now; will start H2 directly after H1 is pushed.

## 2026-11-28 — H1 done (results/hopper/H1/SUMMARY.md)
- 8x H100 80GB SM90, full NV18 mesh, torch 2.14.0+cu132, Triton 3.8.0, FlashInfer 0.7.4, quack-kernels 0.6.5, cutlass-dsl 4.8.0.
- **SM90 default BF16 MoE = `FlashInferCuteDSLSm90Experts` (FlashInfer CuTe-DSL), not Triton.** It is modular, so routing is still vLLM `fused_topk` → P1 applies to the default path too. Working `--moe-backend`: triton, flashinfer_cutedsl, flashinfer_cutlass; trtllm (SM100-only) and batched_triton fail at init.
- QuACK on SM90: gemm, gemm_gated_out swiglu concat B (dense + varlen), varlen gemm all PASS (rel err 1.7e-3). `quack.topk(softmax=True)` PASS for k=8; FAIL for k=4/k=2 at M>1 (mValues stride alignment), as expected.
- Starting H2 (base = default CuTe-DSL and separately `--moe-backend triton`; PR = triton + P1 + P2).
- Node policy from the owner: if anyone else starts using a GPU here, this node steps back immediately (watcher kills my jobs and posts here).

## 2026-11-28 — H2 progress: kernels done, paired E2E running
P1/P2 rebuilt locally on `cfbc42099a` (env-gated, code local). P2 configs pinned from a one-off SM90 autotune at M=4096: **FC1 128x192, cluster 2x1, pingpong; FC2 128x128, cluster 1x2, pingpong** (both non-persistent; all SM90 winners are pingpong). These fixed configs lose ~11–14% on FC1 vs per-M tuning at M=1024/8192.
Kernels (cold L2, CUDA graph, CUPTI, median of 7; `results/hopper/H2/mb_h2.json`):
- **P1** vs `ops.topk_softmax` (E=128, k=8): M=1 1.70x, 8 1.72x, 64 1.59x, 256 1.60x, 1024 1.57x (4096 1.99x, 8192 2.67x). Ids identical, weights within 1.5e-7. The k-derived alignment fixes k=4/k=2 at M>1.
- **P2** vs **default FlashInfer CuTe-DSL** experts: 0.92x / 0.93x / 0.90x / 0.91x / 0.90x / 0.96x at M = 64 / 256 / 1024 / 2048 / 4096 / 8192 (**QuACK slower everywhere**). vs Triton `fused_experts` (no H100 BF16 config file for E=128,N=768 → defaults): 0.88x / 0.93x / 0.94x / 1.01x / 1.10x / 1.18x. Rel err vs FP32: QuACK 0.0029, CuTe-DSL 0.0029, Triton 0.0043.
- Why P2 loses (M=4096, warm, `mb_p2_breakdown.txt`): QuACK GEMMs are faster (FC1 424 vs 433 µs, FC2 228 vs 254 µs), but `moe_permute` + `moe_unpermute` + sort cost ~148 µs vs ~79 µs (FlashInfer gathers inside GEMM1; it pays only unpermute + routing). Same pattern as B14 vs TRT-LLM.
- In-server path check: P1 and P2 both fire (P2 on a 1504-token prefill); greedy output matches.
E2E now running, three paired experiments at once (2 passes x 6 rounds, GPU-swapped): GPUs 0/1 default vs triton+P1+P2; 2/3 triton vs triton+P1+P2; 4/5 (extra) **default vs default+P1**, since P1 also applies to the default path. GSM8K once per arm.

## 2026-11-28 — ack manager update (NUMA split, H4 queued, P1-on-default arm)
- NUMA from `nvidia-smi topo -m` / sysfs: GPUs 0-3 → NUMA 0 (CPUs 0-63), GPUs 4-7 → NUMA 1 (CPUs 64-127); all GPU pairs NV18.
- `numactl` is not installed here; pinning with `taskset -c` (CPU affinity; memory follows by first-touch). Will say so in each result.
- Restarted H2 E2E pinned to NUMA 0 (the first launch was unpinned and spread over GPUs 0-5): GPUs 0/1 **default vs default+P1** (TPOT c1/c8/c32/c128 + TTFT + GSM8K), GPUs 2/3 default vs triton+P1+P2, then triton vs triton+P1+P2 on 0/1.
- H3 kernel sweep running on GPU 6 (NUMA 1); H4 next on GPUs 4-7.
- Step-back watcher: the owner told this session directly to step back if anyone else uses a GPU, so it stays on. It only triggers on GPU usage from other users and never on my own jobs.

## 2026-11-28 — H3 progress: SM90 SwiGLU kernel table (E2E running)
Sweep: QuACK autotune at M=16..8192 for Qwen3-8B (24576x4096) and Qwen3-32B (51200x5120); comparison by CUPTI, cold L2, CUDA graph (`results/hopper/H3/mb_h3.txt`, `mb_h3.json`).
- **One shared 4-bucket table does not fit both models on SM90** (best shared table: mean 4.5%, worst 18% off per-M autotune); 8B wants tile_n 192 and 32B wants 224/160. **Per-weight-shape 4-bucket tables** come within mean 0.8% / 0.6% (worst 4.4% / 4.2%):
  - 8B: M≤128 128x192 c1x2 pingpong; ≤256 128x192 c2x1 pp; ≤384 192x128 c2x1 pp; else 256x192 c1x2 cooperative.
  - 32B: M≤128 128x224 c1x2 coop; ≤256 128x160 c2x1 pp; ≤384 128x256 c1x2 coop; else 256x160 c1x2 coop.
  (All non-persistent; the SM90 search space has no dynamic persistent.)
- vs `F.linear` + C++ `silu_and_mul`:
  | M | 1–8 (F.linear + 2-D Triton) | 16 | 32 | 64 | 128 | 256 | 384 | 512 | 1024 | 2048 | 4096 | 8192 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 8B | 1.02x | 0.99 | 0.98 | 0.99 | 1.05 | 1.09 | 1.06 | 1.11 | 1.15 | 1.10 | 1.10 | 1.08 |
  | 32B | 1.02x | 0.98 | 0.99 | 0.96 | 1.08 | 1.14 | 1.05 | 1.09 | 1.07 | 1.08 | 1.10 | 1.09 |
- **Crossover on SM90 is M≈128** (SM100: 16): QuACK is 1–4% slower at M=16–64, so the E2E PR arm uses F.linear + 2-D Triton SiLU-mul for M<128 and QuACK for M≥128.
- Rel. error vs FP32: fused 0.0017 vs unfused 0.0035–0.0036 (same as SM100). Small-M path (F.linear + Triton) 0.0030.
- Note: vLLM's default here runs `SiluAndMul` as an Inductor-fused Triton kernel (custom_ops none), not the C++ op; the E2E captures that.
E2E running: Qwen3-8B paired on GPUs 4/5 (NUMA 1, taskset); Qwen3-32B after H4's profiling frees GPUs 6/7.
Also ack H5 (permute overhead); will start its kernel work as soon as a GPU frees up.

## 2026-11-28 — H5 done: permute overhead can be cut, but QuACK MoE still loses to the SM90 default (results/hopper/H5/SUMMARY.md)
- Triton index build (cu_seqlens + sorted index + inverse) 14–27 µs vs `moe_permute`; with a Triton row-gather copy, the best permute-free P2 is **0.94x / 0.94x / 0.965x / 0.95x** of the default CuTe-DSL total at M = 256 / 1024 / 4096 / 8192 (current P2: 0.92 / 0.89 / 0.87 / 0.85). Output identical to current P2.
- FC1 with `A_idx` (cp.async gather; SM90 has no TMA gather) is slower than copy + contiguous FC1 even re-tuned (best tile 128x256 cooperative): +78 µs at 4096. Router weight in FC2's epilogue: no gain.
- Below the ≥5% gate → no E2E. Measured on GPU 0 (ratios valid within the table; absolute ~8% high, see below).

## 2026-11-28 — hardware note: GPU 0 is an outlier — excluded from paired E2E
GPU 0 idles at ~71 °C (others ~31 °C), draws ~390 W vs ~335 W on GPU 2 under the same load at the same 1980 MHz with no throttle reason flagged, and is ~8–9% slower E2E: the default arm's TTFT c16 was 1067 ms on GPU 0 vs 981 ms on GPU 2. The first "default vs default+P1" run sat on GPUs 0/1 (its pass-1 TTFT c16 showed −7.5% for P1, which was GPU 0, not P1), so I stopped it and will rerun it on a healthy pair. GPU 0 now only runs single-GPU kernel microbenches (relative numbers). Worth a hardware check by the owner.

## 2026-11-28 — H4 done: QuACK W4A16 beats Machete on H100 after per-shape tuning — passes the gate (results/hopper/H4/SUMMARY.md)
- vLLM uses **Machete** (`MacheteLinearKernel` via AutoAWQMarlin) for Qwen3-8B/32B-AWQ on H100. Machete is 72% / 57% / 87% of GPU time at c1 / c32 / prefill for 8B, and 84% / 73% / 89% for 32B. No ungated Qwen3 GPTQ-Int4 repo exists.
- Kernel (cold L2, CUPTI), per-layer sum of qkv/o/gate_up/down, tuned QuACK vs Machete FP16: **8B 1.50x (M=1), 1.36x (32), 1.11x (512), 1.13x (2048); 32B 1.19x, 1.13x, 1.11x, 1.05x.** QuACK's built-in config heuristic gives much less (8B 1.33x / 1.15x / 0.87x / 1.10x; 32B ~1.0x), so an integration needs a per-shape table.
- Projected step-time saving: 8B ~24% (c1), ~15% (c32), ~10% (prefill); 32B ~13%, ~8.5%, ~4%.
- Caveats: QuACK takes **BF16 activations only** (AWQ checkpoints are FP16 → serve with `--dtype bfloat16`, scales cast to BF16). Error vs FP32: QuACK 0.0023, Machete BF16 0.0032, Machete FP16 0.0003. Needs a repack from AWQ packing: 3–6 ms per layer on GPU, matches vLLM's `awq_dequantize` at 0.0017. Group 128 only for AWQ.
- Ready to run the E2E A/B if you want it (suggested design in the summary).

## 2026-11-28 — ack handover (H4 → hopper2)
- H4 was already finished when the handover arrived; full results are in `results/hopper/H4/` (commit above) for hopper2 to build on. The per-shape tuned table in `sweep_w4.txt` is the key input for an E2E.
- Finishing: H2 E2E — default vs triton+P1+P2 (GPUs 2/3, pass 1 at TPOT), default vs default+P1 (rerun on healthy GPUs 6/7), then triton vs triton+P1+P2 on 2/3; H3 Qwen3-8B E2E (GPUs 4/5). Skipping H3 Qwen3-32B. GPU 0 stays out of paired runs. Final H2/H3/H5 summary after that, then standby.
- **Push blocked**: this node's GitHub token became invalid around 16:20Z ("Invalid username or token"). Commits are queued locally and will be pushed when the owner re-authenticates.

## 2026-11-28 — push restored
Owner re-authenticated; H4 results and the handover ack are now pushed. H2/H3 E2E still running.

## 2026-11-28 17:45Z — H2 E2E default vs PR (triton + P1 + P2): DONE
Qwen3-30B-A3B, GPUs 2/3 (NUMA0, taskset 0-63), 2 passes x 6 rounds, GPU-swapped. Full table: `results/hopper/H2/e2e_def_vs_pr/TABLE.md`.
- **TTFT (4096/1) is slower in every pair:** c1 **+6.4%**, c4 **+7.1%**, c16 **+7.2%** (PR faster in 0/12 for each). This matches the kernel result: P2 loses to CuTe-DSL in prefill because permute/unpermute costs ~148 µs vs ~79 µs at M=4096.
- **TPOT (512/512) is slightly faster:** c1 **−1.7%** (12/12), c8 −1.0% (7/12, noise), c32 **−3.3%** (10/12), c128 −0.4% (7/12, noise). Decode M is below `KOPT_QUACK_MOE_MIN_M`, so this gain comes from Triton decode + P1, not P2.
- GSM8K: base 0.9295, PR 0.9257 (−0.4 pt, within run-to-run noise for 1319 Qs). Greedy: 3/4 prompts identical over 64 tokens, and 1 diverges at token 60 in both passes (BF16 accumulation order, expected).
- Verdict: **the PR as a whole does not beat default.** P2 should not ship on SM90. The decode gain is checked separately by def_vs_defp1 (running on 6/7) and tri_vs_pr (starting now on 2/3).

## 2026-11-28 17:55Z — H3 E2E Qwen3-8B (QuACK fused gate_up + SwiGLU): DONE
GPUs 4/5 (NUMA1, taskset 64-127), 2x6 paired, `KOPT_SWIGLU_SMALL_M=128` with the per-shape SM90 table. Table: `results/hopper/H3/e2e_swiglu_8b/TABLE.md`.
- **TTFT (4096/1) is faster:** c1 **−2.9%** (11/12), c4 **−3.2%** (12/12), c16 **−2.6%** (12/12). This is consistent with the 1.05–1.15x kernel win at large M.
- **TPOT (512/512) is neutral:** c1 +0.1%, c8 −0.1%, c32 +0.9% (2/12), c64 +0.1%, c128 **−1.0%** (12/12). In c1–c64 the sign flips with the GPU swap (GPU 4 vs 5 offset of about ±0.5%), so the paired mean is about 0. Decode M < 128 uses the cuBLAS + Triton SiLU-mul path, which matches the default.
- c32 +0.9% (2/12) is the only cell leaning slower. The 128-row cutoff (`SMALL_M=128`) for the QuACK path is the knob for it.
- GSM8K: base 0.9052, PR 0.9083. Greedy: 4/4 identical over 64 tokens in both passes.
- Verdict: **a clean prefill win (about 3% TTFT), decode neutral, accuracy unchanged.** Qwen3-32B is not run, per the handover.
