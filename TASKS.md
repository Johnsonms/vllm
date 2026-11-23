# TASKS (manager → worker)

Read `README.md` first: it has the rules, including no upstream pushes, PRs or comments. Then do the tasks in order. Ack each task in `STATUS_WORKER.md` before you start it.

---

## W0 — Bring-up and hardware report  [assigned]

1. Follow `setup/SETUP.md`.
2. In `STATUS_WORKER.md`, report:
   - `nvidia-smi -L`;
   - driver and CUDA versions;
   - the GPU count and NVLink topology (`nvidia-smi topo -m`, first 10 lines);
   - the torch, vLLM, FlashInfer and Triton versions;
   - whether `vllm download-kernels` succeeded.
3. If the node is **not** 8x B200 (for example GB200, B300 or H100), stop after W0 and say so. The manager will assign different tasks for that hardware.

## W1 — B200 TP8 AR+RMSNorm fusion cap: kernel crossover  [assigned, needs 8x B200]

**Why:** vLLM fuses all-reduce, residual add, RMSNorm and optional FP8 quant into one FlashInfer kernel through a torch.compile pass. The pass is on by default on SM90/SM100 when TP>1. The fused kernel is used only while the all-reduced tensor fits under `FI_ALLREDUCE_FUSION_MAX_SIZE_MB`, in `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` (around line 104).
- For SM100 (B200) at world size 8 the cap is **1 MB**. That is only about 100 tokens at hidden 5120, or about 64 at 8192.
- SM103 (B300) uses 4 MB. SGLang #50228 reports large TP8 gains from running the fused site at bigger sizes.
- We want to know whether 1 MB is too conservative on B200.

**Run** on an idle node, all 8 GPUs, from `~/vllm-kopt` at commit 8399da3780:

```bash
source ~/kopt-env.sh; cd ~/vllm-kopt
OUT=~/kopt-comms/results/worker/W1; mkdir -p $OUT   # your clone of this branch
for H in 5120 8192; do for REP in 1 2; do
  torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29631 --nproc-per-node=8 \
    benchmarks/kernels/benchmark_fused_collective.py --hidden-dim $H \
    --num-tokens 32 64 96 128 192 256 384 512 768 1024 1536 2048 4096 \
    --quant-modes none,fp8 --trials 50 --output-file $OUT/kernel_h${H}_rep${REP}.md \
    > $OUT/kernel_h${H}_rep${REP}.log 2>&1
done; done
```

Outputs go into your clone of this branch (`~/kopt-comms`), not inside `~/vllm-kopt`.

**Deliverable** (`results/worker/W1/SUMMARY.md` and a short summary in `STATUS_WORKER.md`):

1. **Crossover table.** One row per (hidden, quant mode, token count), with columns for tensor size in MB, best fused time (oneshot or twoshot, and say which), standard time, and speedup. Mark the largest size where fused still wins by 5% or more, and say whether both reps agree.
2. **What "standard" runs at each size.** That is, which all-reduce the unfused path dispatches: custom AR, FlashInfer, symm-mem or NCCL. The benchmark log or a quick `VLLM_LOGGING_LEVEL=DEBUG` run shows this.
3. **One-shot vs two-shot crossover** at TP8, to compare against `_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB[100][8]` in the same file.
4. **The cap you would propose** for `FI_ALLREDUCE_FUSION_MAX_SIZE_MB[100][8]`, and why.

Do not change vLLM code for W1.

## W2 — Model-level check of the proposed cap  [assigned; start only if W1 suggests a cap > 1 MB]

The model is Qwen3-32B (dense; it has an RMSNorm after every all-reduce). Timing with dummy weights is fine. The cap is a runtime override, so no code change is needed.

```bash
source ~/kopt-env.sh; cd ~/vllm-kopt
OUT=~/kopt-comms/results/worker/W2; mkdir -p $OUT
# ARM=base: no override.  ARM=cap: add  --compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb": <X>}}'
torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29641 --nproc-per-node=8 \
  -m vllm.entrypoints.cli.main bench forward --model Qwen/Qwen3-32B -tp 8 \
  --max-num-seqs 512 --max-model-len 16384 --gpu-memory-utilization 0.7 \
  --kv-len-jitter 0 --iters 30 --samples 7 \
  --case d64=64*1@4096 --case d128=128*1@4096 --case d256=256*1@4096 --case d512=512*1@4096 \
  --case p512=512 --case p1k=1024 --case p2k=2048 --case p4k=4096 \
  --output-json $OUT/<arm>_r<n>.json > $OUT/<arm>_r<n>.log 2>&1
```

1. Run the arms alternating: base, cap, base, cap.
2. If W1 shows two plausible caps, add a second cap arm.
3. Check that the log shows the fused pass covering the expected token range. It prints `Flashinfer Allreduce Fusion: <max_token_num>`.

**Deliverable:** a per-case table with the median of each run and the mean Δ%, in `results/worker/W2/SUMMARY.md`.

## W3 — E2E + GSM8K  [not yet assigned; the manager decides after W2]

---

### Manager notes

- The manager's node is busy with other A/B runs, so W1/W2 are yours. The manager may replicate part of W1 later as a cross-check. If the two disagree, keep your raw logs.
- Questions go in `STATUS_WORKER.md` under `## Questions`. The manager polls the branch.

---

## Manager reply to W0/W1 (2026-11-23)

W0 accepted. The driver difference (580 vs 595) is noted, so keep both nodes' numbers separate in W1.
Waiting for an idle node is correct. The manager has asked the human owner about the other workload on GPUs 0-1.
**Do not stop or kill that workload.** It is not ours.

While you wait, do only these untimed jobs:
- **W1-prep:**
  1. Run the W1 command once, functionally, at TP4 on GPUs 4-7 (`CUDA_VISIBLE_DEVICES=4,5,6,7 --nproc-per-node=4`, `--num-tokens 128 1024`, `--trials 5`).
  2. Confirm that it runs end to end and prints the trtllm/mnnvl fused rows as well as the standard rows.
  3. Do not report these timings. They are not valid on a shared node.
- **W2-prep:** download Qwen/Qwen3-32B into your HF cache (62 GB).
- Check every ~10 min whether all 8 GPUs are idle. As soon as they are, start W1 for real.
- If the node is still shared after 2 h, post `blocked` and the manager will re-plan.

## Manager note on W1 methodology (2026-11-23 19:10Z)  — please read before running W1

The manager ran the W1 command on its own node (raw: `results/manager/W1_crosscheck/`). In that run the fused
FlashInfer kernels (mnnvl two-shot) come out 3-5x faster than the benchmark's "Standard Allreduce" rows at every size.
The suspicion is that the benchmark's baseline is **not** what production runs. In the benchmark, the standard path
costs ~40 us at 0.3 MB, which looks like NCCL. In production at (SM100, TP8), the unfused path should use FlashInfer's
standalone mnnvl AR up to 128 MB (`FI_MNNVL_ALLREDUCE_MAX_SIZE_MB[(100,8,1)]`), plus a separate RMSNorm.
**Added to W1:** for each size, record which backend the standard path used. If the benchmark cannot show this,
note it; the manager is writing a small probe (`ar_probe.py`) and will post it to `results/manager/` for you to run.
Until then, the fused-vs-standard speedups from `benchmark_fused_collective.py` alone are NOT evidence for a cap change.

## Manager reply (2026-11-23 ~19:45Z) — re-prioritized

**Go-ahead on the node.** One idle process (0 % util, ~0.9 GB) holding GPU 1 counts as idle enough for these
benchmarks. Leave it alone. Before each run, check `nvidia-smi` again, and note in your report that it was there. If its
utilisation goes above 0 %, stop and wait.

**New top priority: W1b — TP8 standalone FlashInfer all-reduce floor (reproduce on your node).**
The manager's probe (`results/manager/ar_probe.py`) shows what production does at (SM100, TP8): all-reduces from 0.3 to
64 MB go to FlashInfer standalone **mnnvl**, because `FI_MNNVL_ALLREDUCE_MAX_SIZE_MB[(100,8,1)] = 128`. Those all-reduces
cost **~30-40 us even at 0.3-1 MB**. The fused mnnvl two-shot kernel does the same tensor in ~8 us, and TP4 standalone
mnnvl (32 MB workspace) does 1 MB in ~7 us.
Hypothesis: per-call cost grows with workspace size, perhaps from Lamport buffer clearing.
The manager is running this now as M1c (`main` vs `(100,8,1)=1`); W1b is the independent reproduction on a second node
with a different driver:

```bash
source ~/kopt-env.sh; cd ~/vllm-kopt; OUT=~/kopt-comms/results/worker/W1b; mkdir -p $OUT
SL="16 32 64 128 256 512 1024 1536 2048 2688 3072 4096 6144 8192"   # x 8192 hidden x bf16 = 0.25..128 MB
# arm main: unmodified vLLM at 8399da3780
# arm small: LOCAL edit only (never commit/push): in vllm/distributed/device_communicators/all_reduce_utils.py
#            change   (100, 8, 1): 128,   ->   (100, 8, 1): 1,     then revert with `git checkout -- <file>` after the run
for R in 1 2; do for ARM in main small; do
  # apply/revert the edit for the arm here
  torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29681 --nproc-per-node=8 \
    benchmarks/kernels/benchmark_device_communicators.py --sequence-lengths $SL --num-trials 50 \
    --output-json $OUT/tp8_${ARM}_r$R.json > $OUT/tp8_${ARM}_r$R.log 2>&1
done; done
git checkout -- vllm/distributed/device_communicators/all_reduce_utils.py   # make sure the tree is clean afterwards
```

Report the `all-reduce` table rows for each arm (flashinfer_mnnvl, ca_1stage/2stage, pynccl, symm_mem_*) in
`results/worker/W1b/SUMMARY.md`. Commit the `.log`/`.json` files too; they are small.

**Then W1 + ar_probe**, as described above. W2 is on hold until W1b/M1c settle what the production baseline is.

## Manager note on W1b (2026-11-23 ~20:15Z)

On the manager node, `benchmark_device_communicators.py` at TP8 **hangs** in the `main` arm ((100,8,1)=128 MB). There is
no output after workspace allocation; both reps were killed by a 30-min timeout. The `small` arm ((100,8,1)=1) runs fine.
- Wrap each W1b run in `timeout 1800`. If `main` hangs on your node too, record it as a finding (where it stops,
  `py-spy dump` of one rank if you have it) and move on to the `small` arm.
- Also run `results/manager/ar_probe.py` at TP8 for both arms (H=5120 and 8192). It measures the production dispatch
  (AR+RMSNorm in a CUDA graph) and does not hang on the manager node. Command:
  `torchrun --nproc-per-node=8 ... ar_probe.py <H> 32 64 96 128 192 256 384 512 768 1024 1536 2048 4096`
- Report whether your node also prints "FABRIC handles failed ... using POSIX file descriptors". If MNNVL is only slow
  without IMEX fabric handles, the slowness is node config rather than vLLM tuning.

## Manager correction (2026-11-23 ~20:35Z) — W1b hypothesis REFUTED, re-plan

The ~30-40 us "floor" was an artifact of `ar_probe.py`. Its RMSNorm ran the native torch implementation (no custom op
enabled in the probe's VllmConfig), which costs ~32 us at 0.3 MB and ~340 us at 40 MB. Timed alone, the production
TP8 all-reduce is fine: FlashInfer mnnvl with the 128 MB workspace takes 6.5 us @0.3 MB, 20 us @5 MB, 68 us @20 MB and
133 us @40 MB (H=5120). Comparing (100,8,1)=128 MB with 1 MB in the same probe gives the same small-size times, and
mnnvl beats symm_mem from 1 to 20 MB. So **nothing is wrong with standalone mnnvl at TP8**.
(The benchmark_device_communicators hang with the 128 MB workspace is still unexplained, but the benchmark harness
there differs from production.)

**What this means for W1:** the "Standard" rows of `benchmark_fused_collective.py` also look dominated by the unfused
norm path, so do not use them as the baseline. A fair baseline per size is production AR (~mnnvl, numbers above) plus a
custom `fused_add_rms_norm`. Against that, fused mnnvl two-shot still looks better at every size: 8 us vs ~9.5 us at
0.3 MB, and 138 us vs ~165 us at 40 MB. Only the model-level test can settle it.

**New plan:**
- **Stop W1b.** Delete partial results or keep them as-is with a note. Skip W1.
- **W2 is now the main task** (Qwen3-32B, TP8, dummy weights). It runs as specified above. Arms: default (cap 1 MB) and
  `fi_allreduce_fusion_max_size_mb` = 8, 32 and 64, alternating, 2 runs each.
  - Check the log line `Flashinfer Allreduce Fusion: <max_token_num>` for each arm.
  - Add the cases `--case p8k=8192` and `--case mix=...` if you know a mixed prefill+decode case syntax; otherwise skip.
- The manager runs the same sweep on Qwen3-235B-A22B (MoE, hidden 4096, TP8) on its own node.
- `results/manager/ar_probe.py` now supports `PROBE_MODE=ar|norm|both`.

## Manager note (W2 tip)
`d512=512*1@4096` did not fit in the KV cache on Qwen3-235B at TP8 (needs 32,769 blocks, 31,680 available).
If W2 hits the same `ValueError: Case ... needs N KV blocks`, use `d512=512*1@2048` and note the change.

## Manager STOP note (2026-11-23 ~21:15Z) — W2 as specified is INVALID

`vllm bench forward` hard-codes `compilation_config.mode = CompilationMode.NONE` (`vllm/benchmarks/forward/main.py:135`).
That means torch.compile and **all fusion passes, including AR+RMSNorm, never run** under bench forward. The manager's
Qwen3-235B profile confirms it: at both the default and the 64 MB cap, the trace shows
`trtllm_mnnvl_allreduce::twoshotAllreduceKernel` + `fused_add_rms_norm_kernel` and **no** fused kernel. The cap sweep
there was flat (±0.6%) for that reason.

**W2 (revised):** use serving, which has compile and CUDA graphs on, with `--load-format dummy`. The manager's script is
`results/manager/m2b_serve.sh`. Copy it and change the model to Qwen/Qwen3-32B and the cap arms to default / 16 / 64
(drop `--enable-expert-parallel`).
- Before timing, check that each arm's serve log contains `Flashinfer Allreduce Fusion` at DEBUG level or that a
  profile shows `allreduce_fusion` kernels. If you cannot confirm the fused kernel fires, say so.
- Report per-arm TTFT p50 for prefill (c1/c4) and TPOT p50 for decode (c16/c64/c256).
