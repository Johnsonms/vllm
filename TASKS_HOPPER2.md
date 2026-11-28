# Tasks for the second Hopper node (role `hopper2`)

The manager writes this file; the node only reads it. Report in `STATUS_HOPPER2.md` and `results/hopper2/H2-<n>/`.
Read `setup/HOPPER_START.md` and `setup/HOPPER2_START.md` first. Context for the whole round: the "Why Hopper" section of `TASKS_HOPPER.md`.

## H2-0: bring-up  [assigned 2026-11-28]
Env exactly as `setup/HOPPER_START.md` §2 (base `cfbc42099a`, quack-kernels 0.6.5). Record GPU model, driver, versions, `nvidia-smi topo -m`, and per-GPU idle temperature / clocks / a 30 s BF16 GEMM throughput check so we know every GPU is healthy. Push `results/hopper2/H2-0/SUMMARY.md`.

## H2-4: QuACK W4A16 GEMM vs vLLM's int4 kernels (taken over from the first node's H4)  [assigned]
Spec: `TASKS_HOPPER.md`, section "H4". In short: QuACK `gemm_w4` (SM90 weight-only int4 / AWQ / mxfp4 / nvfp4 / fp8 weights) vs what vLLM runs for `Qwen/Qwen3-8B-AWQ` and `Qwen/Qwen3-32B-AWQ` on H100 (Marlin / Machete). Profile shares, kernel microbench at M in {1, 8, 16, 32, 64, 128, 512, 2048}, error, repack needs. Go threshold ≥1.10x on ≥5% of step time → then paired E2E + GSM8K.
If the first node already pushed partial H4 results in `results/hopper/`, start from them instead of repeating.

## Later (the manager will move these over as the first node finishes)
- H3's Qwen3-32B SwiGLU E2E on SM90, if the first node has not run it.
- Any new SM90 follow-ups from the round.

## Update (manager): H4 kernel phase is already done by the first node → your H2-4 is now the E2E phase
Read `results/hopper/H4/SUMMARY.md` and `sweep_w4.txt` first. Result: QuACK `gemm_w4a16(wformat="int4awq")` with a **per-shape tuned config** beats vLLM's Machete on H100: per-layer sum 8B 1.50x (M=1) / 1.36x (32) / 1.11x (512) / 1.13x (2048); 32B 1.19x / 1.13x / 1.11x / 1.05x. Machete is 72-89% of kernel time, so the projected step-time saving is large (8B ~24% at c1). Caveats: BF16 activations only (AWQ checkpoints are FP16), group 128 only, needs a load-time repack (3-6 ms per layer).

**H2-4 (E2E):**
1. Env-gated prototype (code local) in vLLM's AWQ linear path (`AutoAWQMarlinLinearMethod` → `MacheteLinearKernel`): at load, repack each layer's AWQ weights for QuACK (`prepare_w4_weight(..., "int4awq")`); at runtime call `gemm_w4a16` with a fixed per-(N, K, M-bucket) config table from `sweep_w4.txt` (no runtime autotuning; check graph capture works and that a new M does not recompile). Fall back to Machete for anything the table does not cover.
2. Paired E2E, Qwen3-8B-AWQ and Qwen3-32B-AWQ, TP1, 2 passes x 6 rounds, GPU-swapped. **Two baselines**, because QuACK needs BF16:
   - (a) the real default: Machete with the checkpoint's FP16 (no `--dtype`) vs QuACK with `--dtype bfloat16`;
   - (b) like for like: Machete `--dtype bfloat16` vs QuACK `--dtype bfloat16`.
   TPOT 512/512 at c1/c8/c32/c128, prefill TTFT 4096/1 at c1/c4/c16.
3. GSM8K, 4 runs per arm, for (a): FP16 Machete vs BF16 QuACK. BF16 vs FP16 activations is the main accuracy risk.
4. Report: E2E tables, GSM8K, load-time cost, memory, and what landing needs (vLLM kernel class, QuACK changes such as FP16 activation support or other group sizes).
Then, if the first node has not done it: **H3 Qwen3-32B SwiGLU E2E** on SM90 (spec in `TASKS_HOPPER.md` H3; per-shape table in `results/hopper/H3/`).
