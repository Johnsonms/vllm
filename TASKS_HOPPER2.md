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
