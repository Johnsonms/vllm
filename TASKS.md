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

## W3 — Bug hunt: `vllm bench forward --profile-dir` breaks at 128 decode tokens  [assigned 2026-11-23 ~21:30Z]

The node is shared and TP8 timing is not possible, so W2 stays parked until GPUs 0-1 are free. This task is **untimed**
and only needs GPUs 4-7, so it can run now. Before each run, check that 4-7 are idle.

**Symptom** (manager, round 1; Qwen3.8-Flash-Next-FP8, TEP4, MTP3, dummy weights):
- Profiling with `--profile-dir` works at 48-96 decode tokens (~2k GPU activities per replay).
- At 128 tokens it jumps to ~16.4k activities per replay. The total (49,142) is not divisible by 3 replays, so
  `vllm/benchmarks/forward/profile.py:summarize()` raises `ValueError: ... GPU activities don't split into 3 replays`.
- Only that rank raises. The other ranks then hang in gloo until timeout.

**Questions:**
1. **Which activities explode** between d96 and d128? Diff per-name counts per replay. It might be a kernel launched per
   token or per request outside the graph, CUPTI memcpy/memset records, or a fallback to eager. This is the important one:
   **if a kernel really runs ~14k extra times per forward at >=128 tokens, that is a real perf bug in vLLM.**
2. **Why aren't the counts divisible by 3?** Dropped CUPTI records, or an extra activity in one replay?
3. **Why do the other ranks hang?** Is the error raised before a collective the other ranks wait in?

**Setup:**
- `hf download Qwen/Qwen3.8-Flash-Next-FP8 --include "*.json" "*.txt" "*.model" "*.tiktoken" "*.jinja"`.
  With dummy weights no safetensors are needed. If loading complains, report which file it wanted.

**Repro** (GPUs 4-7):
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29701 --nproc-per-node=4 \
  -m vllm.entrypoints.cli.main bench forward --model Qwen/Qwen3.8-Flash-Next-FP8 -tp 4 --enable-expert-parallel \
  --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --max-num-seqs 256 --max-model-len 16384 \
  --gpu-memory-utilization 0.7 --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-len-jitter 0 --routing balanced --iters 10 --samples 3 \
  --case d96=96*1@4096 --case d128=128*1@4096 --profile-dir $OUT
```
- Use `timeout 1800` around the run.
- The raw traces are `$OUT/*.trace.json.gz`.
- Also try without `--speculative-config` and with `--routing zipf`, to see which factor triggers it.

**Deliverable:** `results/worker/W3/SUMMARY.md` with the exploding kernel names and counts, the root-cause hypothesis
with evidence, and the minimal repro. Do not change vLLM code beyond local debugging prints, and revert them afterwards.

## Manager review of W3 (22:03Z) — accepted, excellent work. Follow-up W3b assigned.

**Accepted findings:**
- No kernel explodes.
- The count error comes from a truncated tail of the last replay, and it is flaky.
- The rendezvous all-reduce is counted as a forward kernel (skip is 2 short). This also explains a phantom
  ~600 us `ncclDevKernel_AllReduce_Sum_f32_RING_LL` in the manager's round-1 decode profiles.

**Related open PR:** #62300 "[CI] Avoid nested CUPTI control in bench forward profiles". It stops calling
`cudaProfilerStart/Stop` inside `torch.profiler`, after CI hit empty traces and `bufferCompleted called with
unknown buffer`. Calling `cudaProfilerStop` while torch.profiler still owns CUPTI is a plausible cause of the dropped
tail records too.

**W3b (untimed, GPUs 1-7 are fine).** All edits stay LOCAL on your node. Never commit them to this branch, because
the fork is public. Describe each change in words plus at most a 5-line snippet.

1. **Test #62300.** Run `gh pr diff 62300 --repo vllm-project/vllm | git apply` on top of 8399da3780. Then repeat the
   failing configuration at least 10 cases, for example `--case d129=129*1@4096` x5 and `--case f192=192*1@4096` x5.
   - Use `trace_tail.py` to check record counts and margins on every rank.
   - Question: does truncation disappear, or does the margin improve?
2. **If (1) does not fix it:** keep #62300 applied, add your 50 ms sleep, and repeat the same runs.
3. **Skip fix.** Make `summarize()` split replays at the rendezvous all-reduce, or count the 2 fill kernels in
   `skip`.
   - Confirm that `ncclDevKernel_AllReduce_Sum_f32_RING_LL` no longer appears in `kernels.json`.
   - Confirm that `busy_us` drops by its amount.
   - Find where the 2 fill kernels come from (`capture.py`, probably the flag tensor or event setup) so the
     fix can be exact.
4. **Hang on exception:** if you can, show where rank 3 blocks after raising. Use `faulthandler.dump_traceback_later`
   or `PYTHONFAULTHANDLER`, since py-spy cannot attach on your node.

**Deliverable:** `results/worker/W3/SUMMARY_W3b.md`.

## Manager note (22:23Z) — W2 moved to the manager node

The manager node is free now and the worker's GPU 0 is still shared, so the manager runs W2 itself, using the worker's
`results/worker/W2/w2_serve.sh` unchanged except for paths. Results will appear under `results/manager/W2/`.
**Worker: drop W2 and keep going on W3b.** If W3b finishes before the manager posts W2 results, post W3b and wait for
the next assignment. Do not start W2.

## Manager review of W3b (23:55Z) — accepted

Clear results, and thanks for catching your own invalid runs. The manager is turning (2) and (3) into a local PR
candidate, "U": split replays by `cudaGraphLaunch` correlation plus a short flush sleep inside the profiler, with a
unit test. The manager is running the live A/B on its own node.

**W4 (next for the worker; untimed, GPUs 1-7 are fine): root-cause the `destroy_process_group` hang after an exception.**
- Clean runs exit normally. Find what the exception path leaves alive that blocks
  `GroupCoordinator.destroy -> destroy_process_group` and `ncclCommAbort`. Candidates: an outstanding graph or the
  captured CUDA graph's NCCL work, the benchmark's state views, or the drafter.
- Try a minimal change first: release the graph and state in `run_case`'s `finally` before re-raising.
  Report whether the process then exits.
- Same rules as before: local edits only, describe them in words plus a short snippet, and revert.
- Deliverable: `results/worker/W4/SUMMARY.md`.

## Manager review of W4 (00:52Z) — accepted. Next: W5

W4 is accepted. Fail-fast (C) went into the manager's local candidate U. The manager also reproduced the failure on its
own node with main: 192 requests, 2 ranks truncated, then a 40-minute hang.

**W5: independent hotspot review of Qwen3.8-Flash-Next-FP8 at TEP4 (GPUs 4-7; profiles only, so timing does not need
an idle GPU 0).**
Apply your W3b fixes locally (split by `cudaGraphLaunch` correlation + 50 ms flush) so the profiles are trustworthy.

- Cases: `d1=1*1@4096 d64=64*1@4096 d256=256*1@4096 p2k=2048 p8k=8192`, no MTP, `--routing balanced`.
- For each case, list the top 15 kernels (us, %, calls).
- Map the top ones to Python call sites. Use one eager torch-profiler run with `with_stack=True`, or the
  correlation-to-launch-stack mapping the manager used (`cpu_op` and `python_function` events enclosing the launch).
- Propose at most 3 low-hanging candidates (config/table/small kernel change) that are **not** in this covered list:

  | Item | Status / owner |
  |---|---|
  | skinny-GEMM tables TP2/4/8 | done |
  | hc_down_silu max M | done |
  | shared-expert gate fusion | done |
  | AR+RMSNorm cap TP8 | done |
  | MoE DeepSeek-FP8 activation | fixed upstream in FlashInfer 0.7.6 |
  | QSA prefill sparse kernel | already at DRAM roofline |
  | GDN norm z copy | #50279 |
  | sampler | #66639 |
  | HC MIN_FUSED_UP_M | #69122 |
  | prefill NCCL > 32 MB | M1, shelved |
  | cuBLAS split-K at M=17..256 | noted |

- For each candidate, give the rough per-step share and a 1-line hypothesis. Do not implement.

**Deliverable:** `results/worker/W5/SUMMARY.md`.

## Manager review of W5 (01:17Z) — accepted. Next: W6 (top priority, correctness)

**Thanks. The correctness finding is the most important result.** The manager's own round-1 logs show it too:
`prefill8k ... graph vs eager rel rms 2.7e-02`, while every decode case is exactly 0. Bitwise reruns are not
required, but 1.4-2.7 % relative RMS on the final hidden state is far above normal bf16 summation-order noise
(~1e-3), and it starts above 2,048 tokens. That points to a race, an uninitialized or overflowing buffer sized for
2,048 tokens, or a top-k tie-break.

**Your candidates:**
- **(1) GDN decode state update:** the checkpoint sets `mamba_ssm_dtype: float32`, so the state is fp32. At d256,
  B*HV = 256 x 12 = 3,072, so `BV = 8` (tuned on GB300). That moves 128 KiB per (B, HV) pair, about 403 MB per call, in
  70 us = ~5.8 TB/s, which is already near bandwidth. **Dropped**, except possibly a small B200 BV retune.
- **(2) All-gather copy (1.7 % at d256):** noted for the manager.
- **(3) per_token_group_quant (1-2 %):** noted.

**W6: root-cause the prefill non-determinism above 2,048 tokens. GPUs 4-7, untimed, local edits only.**

1. **Threshold.** Find the exact token count where `eager_repeat` / `graph_vs_eager` leaves 0: try 2048, 2049,
   2056, 2304, 4096. Check whether it depends on `--max-num-batched-tokens`, chunked prefill, or the QSA indexer
   budget (`indexer_budget: 2048` in the config!).
2. **First divergent layer and op.** Run the same restored state through eager twice and compare per-layer outputs
   with forward hooks. Identify the first layer type that diverges (QSA vs GDN vs MoE vs HC) and then the op.
3. **Suspects:**
   - the QSA indexer top-k selection when more than 2,048 candidates exist (ties or an unstable sort; the budget is
     2,048);
   - `_qsa_physical_slots_kernel` / slot layout;
   - FlashInfer GDN chunked prefill (CP kernels with a "Fixup" stage);
   - fp8 KV cache writes racing with reads.
4. **Then decide which it is:**
   - **expected nondeterminism:** an atomic-add reduction, small effect, documented;
   - **a real bug:** wrong data, buffer overflow, race. If so, give a minimal repro.

**Deliverable:** `results/worker/W6/SUMMARY.md`. If you find a real bug, say so in the first line.

## Manager: WRAP-UP (01:25Z) — the round is ending

The human owner has asked to wrap up.
- **Finish W6** (prefill non-determinism above 2,048 tokens) to a clean stopping point, even if the root cause is
  only partial. Post `results/worker/W6/SUMMARY.md` with what is confirmed, what is a hypothesis, and the minimal repro.
- **No new tasks after W6.** After posting:
  - revert every local edit (`git status` clean in your vLLM tree);
  - make sure no GPU processes of yours are left;
  - post a final `STATUS_WORKER.md` entry "W6 done, stood down".
- Keep the raw traces on your node; do not push them here.
Thanks for the work. W3/W3b/W4 became the local candidate U, and W5 found the prefill non-determinism.

---

## W7 — new exploration round (02:28Z, 2026-11-24). The human owner re-activated the worker.

**Context since the round closed:** PR J was opened upstream (#69515). The manager node is re-testing L on the latest
main.

**Constraint:** your GPU 0 is still shared, so W7 is **single-GPU kernel microbenchmarks and profile analysis only**,
on one idle GPU (4-7). Before each timed run, check that the GPU is idle. Run no multi-GPU or serving timing; the
manager does end-to-end validation.

**Rules:**
- Edits are local only and must be reverted afterwards.
- Never push code anywhere. Never comment on GitHub.
- Run a duplicate search for each item and name the closest open PRs.
- **Deliverable:** `results/worker/W7/SUMMARY.md`. Ack in STATUS_WORKER.md first.

### W7a — GDN packed-decode kernel: retune for B200
`fused_recurrent_gated_delta_rule_packed_decode` (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`,
around line 437).
- On SM100 family it uses `BV = 4` if `B*HV <= 64`, else `BV = 8` for an fp32 state. These were "tuned on GB300".
  Qwen3.8 has `mamba_ssm_dtype: float32` and K = V = 128, with HV per rank = 48 / TP: 24 at TP2, 12 at TP4, 6 at TP8.
- The kernel takes 12.4 % of a TEP4 decode step at 256 requests (your W5).
- **Sweep on B200:** B ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} × HV ∈ {6, 12, 24} × BV ∈ {4, 8, 16, 32} ×
  num_warps ∈ {1, 2, 4} × num_stages ∈ {2, 3}.
  - Use the real launcher and patch only BV/warps/stages.
  - Use CUPTI, CUDA graph, cold L2.
  - Check correctness against the current config: outputs and final state, max abs diff.
- Report, per (HV, B):
  - best config vs current, with the speedup;
  - the implied bytes/s (state read+write = B·HV·128·128·4·2 bytes plus q/k/v/out).
- **Question:** is there a ≥10 % kernel gain at common decode sizes, and is a B200-specific branch justified?

### W7b — per-step all-gather plus movedim copy (1.7 % at TEP4 d256)
`tensor_model_parallel_all_gather -> CudaCommunicator.all_gather` runs once per step, followed by a
`movedim` + `reshape` copy when `dim != 0`.
- Find the **call site**: which model/runner code gathers what, with which shape and dim. Likely candidates are logits,
  the sampler, PLE or the MTP path.
- Measure the copy's cost vs batch size (single GPU: time only the copy on the real shape).
- Propose a change that gathers straight into the final layout, and estimate the saving. Do not implement beyond a
  local sketch.

### W7c — `per_token_group_quant_8bit_register_kernel` (MoE input FP8 quant, 1-2 % per step)
- At 8,192 tokens it runs about 26.6 us per call at about 2.4 TB/s (your W5).
- Find how vLLM picks this kernel: `fp8_utils.py::per_token_group_quant_fp8`, the "register" variant vs others. Check
  whether a faster existing variant is available for large N and why it is not chosen.
- Microbenchmark the available variants on the real shape (N × 2560 bf16 -> fp8, group 128, scales in the layout the
  TRT-LLM MoE expects), for N ∈ {1, 64, 256, 2048, 8192}.
- **Question:** is there a ≥1.5x kernel gain at large N without new kernel code (selection or launch config only)?

**Order:** W7a, then W7c, then W7b. Post after each item; do not wait to finish all three.

## Manager review of W7 + decisions (03:52Z)

**W7 accepted.**
- **W7a** (BV=4 on B200, 1.05-1.21x, bit-identical) becomes candidate **V**. It needs model-level evidence; see W8.
- **W7c**: the 2.07x with the float2 -> fp8x2 cvt is noted as a future C++ PR. **Option 1 (source build) is NOT
  approved for now.** The owner decides later.
- **W7b**: duplicate of #65436; follow it.

**Upstream status for your context.** PR J was folded into L at the reviewer's request. #69529 now carries the SM100 TP2 +
TP8 tables, is approved (gau-nernst) and labelled `ready`, with CI running. #69515 is closed.

## W8 — model-level A/B for candidate V (BV=4 for fp32 state on SM100). **Option 2 approved.**
GPUs 4-7 only, TEP4. Check before every timed run that GPUs 4-7 are idle and that GPU 0-3 load is unchanged, and
note in the report what else ran on the node. Edits are local only and must be reverted afterwards.

- **Arms.**
  - `base`: unmodified 8399da3780.
  - `v`: a one-line local edit in `fused_recurrent_gated_delta_rule_packed_decode`, inside the SM100-family branch:
    use `BV = 4` for an fp32 state at every size, and keep `BV = 4` for `B*HV <= 64`. Key it on capability (10, 0)
    exactly, so GB300 is untouched.
  - Describe the edit in words plus at most a 5-line snippet.
- **1. Forward.** `vllm bench forward` (dummy weights; this kernel is not compile-dependent), TEP4, no MTP,
  `--routing balanced --kv-len-jitter 0 --iters 30 --samples 7`.
  - Cases: d32, d64, d128, d256, d512 at @4096, plus mtp_c16=16*4@4096:3 and mtp_c64=64*4@4096:3 with MTP3 (the
    drafter also uses GDN decode).
  - Alternate base, v, base, v.
  - Launch `python -m` with the cwd set to the tree you are timing, and print the module path. The manager learned
    the hard way that a cwd outside the tree silently loads the other code.
- **2. Serving.** `vllm serve ... --load-format dummy` with the same flags as the recipe (TEP4, fp8 KV, fp8 indexer).
  - Decode TPOT at c64 and c256, 512 in / 512 out, `--ignore-eos`.
  - Alternate base, v, base, v.
  - No GSM8K: with dummy weights it is meaningless, and the kernel is bit-identical.
- **Deliverable:** `results/worker/W8/SUMMARY.md` with per-run numbers, mean Δ and spread.
- **Decision rule:** if the serving TPOT gain at c256 is < 0.5 % or inside the noise, report that and stop.

## Manager review of W8 (04:52Z) — accepted, candidate V parked

The A/B is clean: the arm was gated per process and verified, there is a d512 note, and you stopped by the rule.
**Decision: V is parked, not opened.**
- The serving p50 at c256 is inside the noise.
- The mean gain is about 1 %.
- There is no effect under MTP3, and the recipe commonly runs MTP.

The finding is kept in the hand-off for a possible future "B200 GDN decode tuning" bundle.
**No new task. Stand by; you may stop polling.** The manager will post here if anything else comes up.

---

## Round 3 (overnight, 06:11Z 2026-11-24): both nodes look for 5-7 new PRs. Worker: W9, then W10.

**Upstream news:** #69529 (the SM100 TP2+TP8 Qwen4Exp skinny tables from our J and L) **was merged** at 06:01Z.
Table PRs with clean evidence land fast, so this round repeats that pattern where it is still open.

**Finding:** the *generic* BF16 skinny-GEMM table
(`vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py`, `SKINNY_GEMM_PLANS_BY_CAPABILITY`) only has
**SM121** entries. On B200 (SM100), every unquantized BF16 linear of every model without its own table therefore runs
cuBLAS at small M.
- #64979 (open) adds SM90 plans for dense Qwen3 / Llama-3.1 at TP1.
- #67622 (open) adds SM120 plans for GLM.
- **No open PR adds SM100 entries.**

### W9: sweep SM100 generic plans for dense Qwen3 (single-GPU; GPUs 1-7 OK, one sweep per GPU in parallel)
Tool: `results/manager/tools/sweep_skinny_generic.py SHAPES.json OUT.jsonl M...`. It is the manager's sweep
generalised to a shape list; `verify_skinny_reference.py` is the manager's verify script for reference.

1. **Models:** Qwen/Qwen3-8B (TP1, TP2), Qwen/Qwen3-14B (TP1, TP2, TP4), Qwen/Qwen3-32B (TP1, TP2, TP4, TP8).
   - Derive the **local** (N, K) of qkv_proj, o_proj, gate_up_proj, down_proj and lm_head per TP from each
     `config.json`. Only configs are needed.
   - Merge duplicate shapes across models.
2. **Sweep** M ∈ {1, 2, 4, 8, 16} for every shape.
   - Keep only entries that beat cuBLAS by **> 5 %**, the generic table's rule; prefer ≥ 10 %.
   - Drop M=16 unless the gain is ≥ 10 %. On Qwen4Exp, M=16 entries regressed the full forward.
3. **Verify** the kept entries.
   - 4 processes on 4 idle GPUs, 3 interleaved rounds (cuBLAS vs plan), CUPTI, CUDA graph, cold L2.
   - Report the median and the max spread.
   - Record max abs error vs an fp32 reference for both cuBLAS and the plan.
4. **Deliverable:** `results/worker/W9/SUMMARY.md` with the proposed `SM100_SKINNY_GEMM_PLANS` entries as text
   (shape → {M: config}), a per-model/TP table of kernel speedups, and the shapes that should NOT get entries.
   - Do not edit vLLM files. The manager writes the table and runs the E2E.
   - Post after the sweep and again after the verify.

### W10 (after W9): same for the Qwen3 MoE family
Qwen/Qwen3-30B-A3B (TP1, TP2) and Qwen/Qwen3-235B-A22B (TP4, TP8, with EP).
- **Shapes:** attention qkv/o, router (replicated), LM head. The experts are FusedMoE and not in scope.

### Manager side, in parallel
- Open U (bench-forward profiler fix; retest running).
- Prepare B (Qwen4Exp TP4 router/LM-head plans).
- Run E2E for W9's entries on Qwen3-32B (TP1/2/4/8, real weights on the manager node) and Qwen3-8B.

---

## Round 3 outcome (manager, 11:00Z): W9 and W10 were done on the manager node

The worker had stopped polling, so the manager ran W9 and W10 itself. Results are in `~/kopt-logs` on the manager node, summarised here:
- **W9 (dense Qwen3):** 41 shapes, 91 entries, 1.10–2.59× over cuBLAS. E2E TPOT at c1 −14 to −17.5%, at c2 −10 to −15% (Qwen3-8B TP1; Qwen3-32B TP1/4/8). GSM8K unchanged.
- **W10 (Qwen3 MoE attention/router/LM head):** 49 entries. Qwen3-30B-A3B TP1: c1 −10.1%, c2 −10.4%. Qwen3-235B TP8: c1 −11.0%.
- Both are drafted, awaiting the owner's OK.

**Gotcha for any serving A/B of a plan table:** the torch.compile AOT cache key does not include `skinny_gemm_plans.py`.
- The second arm silently loads the first arm's graph, which has the use-skinny decision baked in.
- Always export `VLLM_DISABLE_COMPILE_CACHE=1`, and check that `grep -c 'Directly load AOT' serve.log` is 0.

## Round 4 (morning, 2026-11-24): the owner asked both nodes to keep finding opportunities. Worker: W11.

### W11: SM100 generic skinny plans for Llama-3.x, Qwen2.5 and Mistral-Small (single-GPU sweeps; GPUs 1-7 OK)
Same method as W9: tool `results/manager/tools/sweep_skinny_generic.py`, then a 4-seed verify with CUPTI.

1. **Models.** Configs only. The meta-llama and gemma repos are gated for us, so use the ungated mirrors listed here.
   - `unsloth/Llama-3.1-8B-Instruct` (TP1, TP2)
   - `unsloth/Llama-3.3-70B-Instruct` (TP2, TP4, TP8)
   - `Qwen/Qwen2.5-7B-Instruct` (TP1, TP2)
   - `Qwen/Qwen2.5-72B-Instruct` (TP4, TP8)
   - `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (TP1, TP2, TP4). Text model only; check whether its config nests `text_config`.
2. **Shapes.** Local (N, K) of qkv, o, gate_up, down and lm_head.
   - **Skip any shape already in `SM100_SKINNY_GEMM_PLANS`** on branch `kopt/sm100-skinny-qwen3-moe`. The manager will post that list as `results/manager/W11_existing_shapes.json`.
   - Qwen2.5 has a qkv bias. Note whether the skinny path accepts a bias (check `use_skinny_gemm` / the linear method). If it does not, skip the qkv shapes of Qwen2.5.
3. **Sweep** M ∈ {1, 2, 4, 8}. Keep ≥ 10%.
4. **Verify** as in W9: 4 seeds, CUPTI, CUDA graph, cold L2. Keep entries whose median is ≥ 1.10×. Record the error vs an fp32 reference for both cuBLAS and the plan.
5. **Deliverable:** `results/worker/W11/SUMMARY.md`, plus `proposed.json` and `verified.json` in the same format as the manager's `w9/verified.json`: `{"N,K": {"M": "SkinnyGemmConfig(...)"}}`.
   - Post after the sweep and again after the verify.
   - Do not edit vLLM files. The manager writes the table and runs the E2E.

### Manager side, in parallel
- Profile decode after Q3D+Q3M (Qwen3-32B TP1 and Qwen3-30B-A3B TP1, c1/c8) to find the next non-GEMM hotspot.
- Download the Llama-3.1-8B and Qwen2.5-7B weights for W11's E2E.

### W11 REVISED (owner, 11:25Z): focus on newer models. Drop Llama-3.x, Qwen2.5 and Mistral-Small.
The SM90 PRs below show where B200 has no plans yet. For each, mirror the SM90 shape list on SM100.
The new entries go into the same `SM100_SKINNY_GEMM_PLANS` as Q3D, so the PRs will stack on it.

**W11a (worker): Qwen3.5 / Qwen3.6 MoE.** Mirrors #64982.
- **Models:** `Qwen/Qwen3.6-35B-A3B-FP8` (TP1, TP2, TP4) and `Qwen/Qwen3.5-397B-A17B-FP8` (TP4, TP8). Configs are reachable; the architecture is `Qwen3_5MoeForConditionalGeneration`.
- **Shapes:**
  - The BF16 layers that stay unquantized in the FP8 checkpoint: the GDN fused b/a projection, the shared-expert gate, the router and the LM head.
  - List the layers a module walk reports as unquantized. Load with `--load-format dummy` and print `UnquantizedLinearMethod` layers with their local (N, K).
- **Check how #64982 handles N=1.** The `(1, 2048)` shared-expert gate needs an eager fast path in `layers/utils.py`.
  - Report whether the SM100 path reaches the skinny kernel for N=1 without that change.
  - If it does not, keep N=1 out of scope and say so.
- **Sweep, verify and deliverable as in W9:** M ∈ {1, 2, 4, 8}, ≥ 10%, 4-seed CUPTI verify, and `results/worker/W11/{SUMMARY.md, proposed.json, verified.json}`.
- If time remains, profile one decode step of Qwen3.6-35B-A3B-FP8 at TP1 on B200 with dummy weights. List the top 15 kernels at c1 and c8.

**Manager side:** GLM-5.3-Flash SM100 plans, mirroring #65306 (SM90, TP4/TP8), with a dummy-weight E2E. Also profiling of new-model decode on B200.

## Manager review of W11a (19:2xZ): accepted

Thorough work. The module walk, the N=1 reachability check with matching greedy tokens, and the correction on the quant kernel are all useful.
- **Table:** all 49 entries were applied as-is. Branch `kopt/sm100-skinny-qwen36`, commit 9b861e6c78, stacked on the GLM-5.3-Flash entries (839c887f40) and Q3M (744ac0d84f).
  - Pre-commit passes.
  - `tests/kernels/test_bf16_skinny_gemm.py`: 531 passed; plans_are_valid 234 passed.
- **Next on the manager side:** E2E of Qwen3.6-35B-A3B-FP8 TP1 with real weights (downloaded) plus GSM8K, base 839c887f40 vs 9b861e6c78, after the GLM A/B frees GPUs 0-3.
- **Your quant finding changes priorities:** 200 launches at about 2.5 µs each is a launch-count problem, so W7c's conversion speedup matters little at decode. Taken as the next task.

## W12 (worker): fewer `per_token_group_quant` launches in Qwen3.6-35B-A3B-FP8 decode on B200

There are 5 quant launches per layer, 200 per step: about 10% of the step at 1 request and 9% at 8.
1. **Map each launch to its producer** (e.g. RMSNorm → qkv / GDN in_proj, attention or GDN output → o_proj, silu_mul → down, ...). Use the module walk or the profiler's Python stacks with dummy weights.
2. **For each, find whether vLLM main already has a fused kernel and a compile pass for it, and why it does not fire here.** Candidates:
   - RMSNorm + per-token-group FP8 quant fusion (`RMSNormQuantFusionPass` and friends);
   - silu_mul + group quant;
   - quant folded into the GDN output gated-RMSNorm.
   Check `PassConfig` flags (`fuse_norm_quant`, `fuse_act_quant`, ...) and their SM100 / block-FP8 gating.
3. **If a flag or a small gating change enables a fusion,** run a serving A/B with dummy weights: TP1, c1/c4/c8, 512/512, alternating ×2, `VLLM_DISABLE_COMPILE_CACHE=1`, and check the AOT count is 0.
   - Do not use `bench forward` for this: it forces `CompilationMode.NONE`, so compile passes never run.
4. **Deliverable:** `results/worker/W12/SUMMARY.md` with the launch → producer table, the fusion status per launch and the A/B if run.
   - Post after step 2 so the manager can decide before you spend GPU time.
   - Local edits only; no vLLM commits.
- **Dup check:** search open PRs for "group quant fusion", "fuse_norm_quant", "per_token_group_quant fusion" and "block fp8 fusion SM100" before proposing anything.

## Manager decision on W12 (20:xxZ): option 1 approved

An excellent map. The mode-NONE finding (#51012) matters beyond this task. It also explains why compile-cache effects do not apply to this model.
- **W12b (worker):** prototype launch 4 locally. Make `CutlassFp8BlockScaledMMKernel.input_quant_key()` return `kFp8Dynamic128Sym`, and have `apply_weights` consume the `QuantizedActivation`.
  - Agree on one scale layout between the producer and the kernel. Check M = 1, 2, 4, 8 and a prefill size, e.g. 2048, where the layouts differ.
  - **Correctness:**
    - Unit-level: fused path vs unfused path output, max abs diff, at those M values.
    - Greedy tokens equal on 2 prompts, real weights if you have them; otherwise state dummy-only.
    - Run the existing tests for `fused_act_quant` and block-FP8 linears.
  - **Serving A/B:** Qwen3.6-35B-A3B-FP8, TP1, dummy weights, c1/c4/c8, 512/512, alternating ×2.
  - Also check that no other model that uses this kernel changes behaviour unintentionally. List which `maybe_fused_act_quant` / `_norm_quant_key` call sites would now start fusing.
  - Local only; post a summary with the diff described in words plus at most a 15-line snippet.
- **Launch 1 / option 2:** deferred until 4 is measured.
- **Manager:** takes the sigmoid × mul pair (section 4). The manager has a local Triton kernel for exactly `sigmoid(gate) * out` from earlier work, and will A/B it on Qwen3.6.

## Manager note for W12b (21:xxZ): check the shared-expert stream first; launch 4 may be off the critical path

The manager's S A/B on an idle node (Qwen3.6-35B-A3B-FP8 TP1, real weights) shows **zero** E2E change at c1-c8, within ±0.2%.
- The fused sigmoid-gate kernel does run: 9,920 fused calls, no fallback.
- It is 3.5-4.3x faster than gate + sigmoid + mul at the kernel level.
- **Reason:** `runner/shared_experts.py` runs the whole shared expert on an aux CUDA stream, overlapped with the routed experts (`MULTI_STREAM_OVERLAPPED`), whenever tokens ≤ `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`. At decode, shared-expert time hides behind the routed MoE.

**Launch 4 (shared-expert `down_proj` quant) is in the same shared expert,** so it likely has the same problem. Before the serving A/B:
1. Confirm the order this model takes at decode (`SharedExpertsOrder`) and the threshold value.
2. Quick check: run the prototype with `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` in both arms as a diagnostic, and with the default in both arms as the real case. If the gain appears only with the stream disabled, report that and stop.
3. If launch 4 is off the critical path, say so. Then propose which of the **main-stream** launches (1 attention input, 2 o_proj/out_proj, 5 routed-experts input) has the cheapest fusion, without starting it.

## Manager review of the W12b interim (20:3xZ)

A careful prototype. The scale-layout table and the who-else-fuses list are exactly what a reviewer will ask for. Before any decision, four things are needed:
1. **The idle-node A/B you are running.** The first run overlapped the test suite on another GPU. This model is host-sensitive: the manager saw a fake 6% gain from a contended run. Report c1/c4/c8 from the idle run only.
2. **Reconcile with the aux-stream finding.** S (gate + sigmoid + mul fused, 80 fewer launches per step) gave zero E2E on an idle node, because the shared expert runs on the aux stream at decode. Your change is also inside the shared expert, yet shows −5% at c1.
   - Check which `SharedExpertsOrder` this model takes at 1 token on your node.
   - Run both arms once with `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` as a diagnostic.
   - If the gain survives with the stream on, explain why: for example, the shared expert is the longer branch at c1, or the routed MoE waits on the event.
3. **Accuracy.** Since the outputs are no longer bit-identical, run GSM8K with real weights on both arms (`tests/evals/gsm8k/gsm8k_eval.py`, chat completions, `enable_thinking: false`, max tokens 1024, temperature 0, concurrency 64). Report both arms twice.
4. **Possible latent bug on main.** You found that the registered SiLU-mul block-quant producer writes row-major scales, which is wrong for a column-major consumer from 2 tokens up.
   - Is that producer reachable **on main today** with a consumer that needs column-major scales? Candidates: modelopt, online FP8, compressed-tensors W8A8 block, Quark with a block kernel, or a DeepGEMM-based kernel that exposes `kFp8Dynamic128Sym`.
   - If yes, that is a correctness bug worth more than the perf change. Give a minimal reproduction: a model or quant config plus the observed error, with no code pasted here.
- **Revert your local edit when done**, as you planned. No code in this branch.

## Manager review of W12b final (21:1xZ): strong result. One discrepancy to settle first → W12c

Accepted as a candidate:
- −5.1 / −2.2 / −3.7% TPOT on an idle node; GSM8K indistinguishable.
- Clean answers on the stream order, and the latent-bug check (not reachable on main; it is a trap for the first consumer).

**The discrepancy:** your arms run on 8399da3780, where base TPOT at c1 is **3.54 ms**. The manager's S and #69736 runs use **e66bede56b**, where the same model and flags give **3.23 ms** at c1 (B300 gives 3.00). Your trace shows the shared expert as the longer branch by about 11 µs per layer. If newer main shortened the shared branch, or lengthened the routed one, the lag may be gone at e66bede56b. That would explain why S gave zero there and would shrink W12b too.

### W12c
1. **Rebase the prototype locally** onto `e66bede56b`, the round's current base (no C++ changes since the wheel), and check that the base reproduces about 3.23 ms at c1.
2. **Idle-node A/B** with real weights, TP1, c1/c4/c8, 512/512, alternating ×2, compile cache off.
3. **One profiled decode trace of the base at e66bede56b** (c1): measure the same shared-vs-routed end lag per layer that you measured on 8399.
4. **If the gain holds at e66bede56b,** also run one A/B at c1/c8 with `--max-num-seqs 256` on Qwen3.6 at TP2 (2 GPUs). That shows the effect is not TP1-only.
5. **Deliverable:** append to `results/worker/W12/SUMMARY.md`. Keep the diff on the node; no code here.

If the result holds, the manager will write it up as a PR candidate for the owner. The candidate would combine three parts: the block-FP8 kernel accepting a pre-quantized group-128 input, the `fp8` method exposing its key, and the producer's column-major scales, all in one change.

## Manager note (11-25): W12c is upstream as #69778
Re-implemented on main fb159244e6 (block-FP8 layers only for the `fp8` key exposure; producer writes unpadded column-major scales). Manager paired A/B with GPU swap: −3.8 / −2.5 / −2.4% TPOT at 1 / 4 / 8 requests, 36/36 pairs. GSM8K 10 runs per arm: 0.9636 vs 0.9595; per-question flips are the same near-ties as within-arm (no systematic loss). Thanks — your paired-A/B method and the scale-layout finding are both in the PR text. No new W-task yet; keep polling.

## W13 (11-25): shared-expert aux-stream starvation on B200 (same experiment as B300's B7)
Read `results/b300/B6/SUMMARY.md` first. On B300, at c ≥ 4 the shared expert on the aux stream waits ~11 µs per layer behind the routed-MoE GEMMs (it is the longer branch, as your W12 trace showed on B200).
- Do **B7 steps 1–4 from `TASKS_B300.md`**, on B200, with the same method and arms. Base = main at `04f0149d47` (it does not yet include #69736's table; note that in the report).
- If #69778's fusion is easy to stack, add one arm with it. Optional.
- Deliverable: `results/worker/W13/SUMMARY.md`. Describe edits in words only.

## Manager review of W13 interim (11-25): very useful. → W13b
- The B200 result (−6.7% at c4, no regression at c8/c16) differs from B300's B7 (c4 **+2.5%**, c16 +1.1%). The likely reason is the arm design. B7 had #69736's skinny table on in both arms, which already lets the shared expert overlap at c4. Your base has cuBLAS GEMMs. A P PR would land after #69736, so **W13b: repeat the paired A/B of P with #69736 in both arms.**
  - #69736 is public: `https://github.com/Johnsonms/vllm.git` branch `sm100-skinny-qwen35-moe`, head `ce7e3d5dd7` on main `04f0149d47`.
  - c = 1, 2, 4, 8, 16. Same paired method with the GPU swap; greedy check.
  - If P still wins at c4 without regressing anywhere, also run c32 and c64, plus one other shared-expert model if you have one: Qwen3.5-397B-A17B-FP8 at TP4/TP8, dummy weights are fine for TPOT.
- **Your GSM8K note matters for #69778.** P is output-identical at 1 request, yet GSM8K at 64 concurrency moved 0.9629 / 0.9636 → 0.9591 / 0.9591. That is the same −0.4 pt shift #69778 shows. If you have time, dump per-question correctness for P vs base (wrap `_score_gsm8k`) and report the flips. That is direct evidence that a timing-only change moves GSM8K by this much on this model.
- Drop the optional W12c+P arm; it is not needed now.

## W14 (11-25, overnight): next opportunities on the newest models — profile and rank, no code edits
The owner says **6 GPUs** on your node are available to us (the owner's own job keeps the rest; never touch its GPUs). W13b keeps priority; use the remaining GPUs for W14 in parallel. The manager is covering Qwen3.6-35B-A3B on its own node, so **do not profile Qwen3.6**.

Code under test: #69736 (approved, in CI), `https://github.com/Johnsonms/vllm.git` branch `sm100-skinny-qwen35-moe`. **Pull it; the CI bot merged main into it as `740ff0d02c85`, so use that head.** Set `PYTHONPATH` per tree and log the loaded file path, as before.

1. **Qwen3.8-Flash-Next-FP8, TEP4** (4 GPUs). Use the recipe flags: `--enable-expert-parallel --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --language-model-only --max-model-len 32768`, no speculative decoding. Download by name if missing.
   - Decode profile at c1, c4, c16 (B1/B6 method).
   - Report: top 20 kernels by time share per step; launches per step by kind; cuBLAS BF16 GEMMs still present (shape, M, kernel name); FP8 quant launches and their producers; small elementwise or norm kernels that run back to back; and the busy-union vs TPOT check.
2. **Qwen3.5-397B-A17B-FP8, TP4**, dummy weights if the real ones are not on the node. Same profile at c1 and c8.
3. **Rank the three best candidates** by expected TPOT gain (µs per step ÷ TPOT). For each, give: what would change; which upstream PRs already touch it (search open PRs); whether the output stays bit-identical; and a rough size of the change.
   - Exclude: group-128 RMSNorm + FP8 quant fusion (owner-local work on another node), SiLU-and-mul + FP8 quant (#69778), and skinny-GEMM plans that #69736 already covers.
4. **Deliverable:** `results/worker/W14/SUMMARY.md` (kernel tables < 5 MB each). Post an interim after step 1.

## Manager note on W14 interim (11-25): good. Two things for the ranking.
1. **Block-FP8 linear GEMM (397B: 27.7% at 9 µs per call) is the manager's.** The manager found that FlashInfer's `gemm_fp8_nt_groupwise(..., backend="trtllm")` is **bit-identical** to vLLM's `cutlass_scaled_mm` blockwise path and 1.03–1.54× faster at small M. It was swept over 45 N×K pairs; the rule is `M*N <= 2^18`, except K ≥ 7168 with 6144 ≤ N ≤ 8192. Qwen3.6 E2E is running. Do not duplicate this; just list the 397B shapes and calls per step it would cover.
2. **The Qwen3.8 skinny-table gaps** ((10240, 320) at 97 calls/step, (1, 2560), (2560, 160), and larger M) fit the "single follow-up small-GEMM PR" promised on #69736. Rank them, and if they come out on top, **sweep + verify them as W15** (same method as W11/B2: sweep, then a 4-seed × 3-round verify; keep entries ≥ 1.10×). You can start that without waiting for the manager.

## W16 (11-25): end-to-end check of the W15 Qwen3.8 skinny table entries
W15 is good work: 32 of 35 entries ≥ 1.10× on all 4 seeds. Before any PR we need E2E, because in #69736 part 5 the router M4/M16 and LM-head M4 entries won at kernel level and lost in the full forward. You can start now.

**Arms**
- **Base:** current upstream main (it already contains #69736). Record the SHA.
- **Candidate:** base plus the 32 `verified.json` entries added to the Flash-Next SM100 table. Local commit only; no push.
- In each server log, log the loaded table file path and HEAD. Also log how often each new (N, K, M) entry is hit during one decode step at c1, c4 and c16, so we know the entries actually fire.

**Model and setup**
- Qwen3.8-Flash-Next-FP8, TEP4, with the W14 recipe flags and no speculative decoding.
- `VLLM_DISABLE_COMPILE_CACHE=1`; the AOT count must be 0 in every server log.
- Random 512 in / 512 out with `--ignore-eos` and no prefix caching.

**A/B method.** Two TEP4 servers need 8 GPUs, and you have 6, so use interleaved restarts on the same 4 GPUs (2–5):
- Run A B A B A B (at least 3 server starts per arm).
- After each start, warm up, then run 4 bursts per concurrency.
- Report, per arm, the median TPOT of each start and the spread across starts.
- Report the Δ as the mean of the per-start medians. Count a gain as real only if it exceeds the base start-to-start spread.
- Do not use one start per arm: that is the sequential A/B we have rejected before.

**Concurrency.** c = 1, 2, 4, 8, 16. Also run c32 as a no-regression check; no entry covers M > 16, so it should be flat.

**Correctness**
- Greedy check: 32 single prompts + 256 batched prompts × 256 tokens, base vs candidate, plus a base-vs-base run as the floor.
- If not identical, run GSM8K (5-shot, 1319 q) for at least 8 runs per arm, with per-question dumps and the base-vs-base flip floor.

**If the E2E gain is under about 1% at every c, or any c is slower**
- Ablate by layer group: (a) shared-expert gate (1, 2560); (b) router (512, 2560); (c) (24, 2560) + (320, 2560); (d) large projections M3; (e) LM head.
- Report which groups carry the gain, so the PR keeps only those.

**Dup check.** Search open PRs for "skinny", "Qwen3.8" / "Flash-Next" and "SM100 small GEMM" before writing up.

**Deliverable.** `results/worker/W16/SUMMARY.md`, with the entries described in words or JSON only (no code patches). Post an interim after the c1–c4 A/B.

GPUs 0–1 stay with the owner.

## Manager on W16 interim (b4937bb509)
Good interim: an honest negative at c16, and the hit counting found real issues. Notes for the rest of W16:
1. **Drop the 4 M = 7 entries.** They never fire.
2. **The ablation is the right next step.**
   - Suspects for c16: the router (512, 2560) at M = 16 (1.20× kernel-level, the weakest) and the gate (1, 2560) at M = 16. #69736 part 5 lost the router M16 the same way.
   - If one group explains it, drop that group, or just its M = 16 line, and re-run c1–c32 with the final set: 3 starts per arm.
3. **Sweep M = 10 and 14** for the shapes that fire there, only if the final set keeps those shapes. Same method as W15 (4 seeds, ≥ 1.10×). Optional.
4. **Batched greedy on TEP4 is not reproducible** (floor 189–251/256 differing). Use the single 32 (base floor 0/32) plus GSM8K 8+8 on the final set, as you planned. Note in the summary that expert-parallel batched greedy is nondeterministic here.
5. **#69907 (FlashInfer cute-dsl as the default small-batch BF16 GEMM) is open.** Check whether it routes these same layers away from the skinny table. If it does, our table lines may be dead once it merges; say so in the summary.

## W21 (2026-11-28, high priority): QuACK FP8 kernels vs vLLM's FP8 defaults on B200 — kernel scouting
Context: the round's QuACK work so far (see `TASKS_HOPPER.md` "Why Hopper") found that QuACK's **gated GEMM (SwiGLU in the epilogue) beats cuBLAS + silu_and_mul by 2-3.5% E2E** on BF16 dense models; a PR is ready. Production models are mostly FP8, so the next question is whether QuACK's FP8 / blockscaled kernels beat what vLLM runs for FP8 today. quack-kernels 0.6.5 (installed with vLLM) has `quack/blockscaled/` (mxfp8, mxfp4, nvfp4 operands), `gemm_interface.gemm_quant_out` and `epilogue/quantize_out.py` (GEMM whose output is written already quantized, with blocked scale factors).
**Phase 1 is kernels only, no vLLM edits.**
1. For `Qwen/Qwen3-8B-FP8` and `Qwen/Qwen3-32B-FP8` (block-128 FP8): from a short serve profile, record exactly which kernels vLLM runs on B200 for qkv/o/gate_up/down (CUTLASS block-FP8, DeepGEMM, FlashInfer, ...) and for the activation quant (`per_token_group_quant`, `silu_and_mul` + quant fused or not), with their time share at c1/c32/prefill 4096.
2. Check whether QuACK can consume these checkpoints' scaling as is (block 128x128 weight scales, 1x128 activation scales). QuACK's blockscaled path is MX-style (1x32 UE8M0) and nvfp4; if the formats are incompatible, say so plainly. That is a finding.
3. Where the formats work (or with an MXFP8 re-quantization of the weights, clearly labeled), microbench (cold L2, CUPTI, CUDA graph) QuACK vs vLLM's kernel at M in {1, 8, 32, 128, 512, 2048, 8192} for: plain GEMM; gate_up gated GEMM with SwiGLU **and FP8-quantized output** (`gemm_quant_out` / quantize_out epilogue) vs vLLM's gate_up GEMM + `silu_and_mul` + quant. Report the speedup and relative error vs an FP32 reference.
4. Deliverable `results/worker/W21/SUMMARY.md`: a ranked table of (op, M range, speedup, format caveat, estimated share of step time). Go threshold: ≥1.10x at an M range that carries ≥5% of step time. No vLLM edits, no GitHub actions.

## W22 (2026-11-28, queued after W21): fold residual-add + RMSNorm into QuACK GEMM epilogues (Qwen3 dense BF16, B200)
Profile (`results/manager/quack2/q3_8b_decode_profile.txt`, Qwen3-8B, c1/c32): Inductor's `fused_add_rms_norm` kernels (2 per layer, ~4 + ~2.6 us) are ~5-7% of kernel time at c1 and ~4.6% at c32. QuACK 0.6.5's epilogue library (`quack/epilogue/library.py`) has the pieces to remove them:
- `rms_partial_epi` (o_proj / down_proj GEMM writes `resid_out = acc + residual` and a per-row sum-of-squares reduce);
- per-row `rstd` applied in the **next** GEMM's epilogue (`norm_swiglu_mod(acc, rstd, bias)`, or any mod with an rstd colvec): RMSNorm(x) @ W = rstd(x) * (x @ (diag(w_norm) W)), so fold the norm weight into the next GEMM's weight offline and apply rstd per row in its epilogue.
1. Microbench on Qwen3-8B and Qwen3-32B shapes at M in {16, 32, 64, 128, 512, 2048, 8192}: (a) today: cuBLAS o_proj + Inductor fused_add_rms_norm + cuBLAS gate_up (+ silu); (b) QuACK o_proj with `rms_partial_epi` -> QuACK gate_up gated with rstd applied in the epilogue (combine with SwiGLU; check whether such a mod exists or write one with `@gemm_epilogue`). Same for down_proj -> next layer's qkv_proj. Report the time of the whole chain, error vs FP32 reference, and any QuACK limitation (e.g. the reduce needing a full-row tile, split-N).
2. Below M=16 vLLM uses its CuTe skinny GEMM (66.8% of c1 kernel time), so focus on M>=16.
3. Deliverable `results/worker/W22/SUMMARY.md` (go threshold: chain ≥1.10x at M>=32). No vLLM edits yet, no GitHub actions.

## Manager review of W21 + correction (accepted) → W23 queued after W22
Excellent work, and thanks for the fast correction. `swiglu_quant_mod` giving **1.12-1.44x for the whole gate_up chain (MXFP8 in, SwiGLU, MXFP8 out)** is the strongest QuACK kernel result of the round. The tile_n=192 quantized-output bug goes on the QuACK upstream list (keep the repro).
Open questions that decide whether it can land: the format (block-128 checkpoints → MXFP8 re-quant) and who consumes the MXFP8 output (DeepGEMM's down_proj expects 1x128 packed scales, not 1x32).

## W23 (queued after W22): end-to-end MXFP8 MLP prototype on Qwen3-8B-FP8 (B200)
Goal: find out whether the kernel win survives end to end and what it costs in accuracy.
1. Env-gated prototype, code local. At load, re-quantize gate_up and down weights of every MLP from the block-128 checkpoint (dequantize → MXFP8 1x32, UE8M0), interleaving gate/up rows for the gated kernel.
2. Forward for M ≥ a threshold you pick from W21 (start at 32): activation MXFP8 quant (find the fastest available kernel: vLLM / FlashInfer `mxfp8_quantize`, or fold it into the preceding RMSNorm if cheap) → gate_up via `swiglu_quant_mod` (MXFP8 out, configs pinned, **never tile_n=192**) → down_proj via QuACK MXFP8 plain GEMM. Below the threshold, keep vLLM's DeepGEMM path (keep both weight copies for the prototype; report the memory cost).
3. Paired E2E as usual (2 passes x 6 rounds, GPU-swapped): TPOT c1/c8/c32/c128 and prefill TTFT 4096/1 c1/c4/c16. GSM8K **4 runs per arm** (re-quantization changes the weights, so accuracy is the main risk).
4. Report: E2E table, GSM8K, memory, and the list of QuACK / vLLM changes a real landing would need (e.g. QuACK support for 1x128 / 128x128 FP8 scales, which would remove the re-quant).

## Manager review of W23 (accepted) → W24
Strong result: prefill TTFT -4..-8 % (35/36), decode -1.4 / -2.8 % at c32 / c128 on Qwen3-8B-FP8. Two loose ends before this goes into the landing plan:
1. **Decode gap** (kernel estimate ~7 % at c32, E2E -1.4 %): profile v2 at c32 (base vs prototype, same GPU) and account for the difference per kernel (warm L2 effects, launch gaps, the activation-quant kernel, any copy / layout op like the one I found in my QKV prototype, where attention silently `.contiguous()`-copied a strided q).
2. **Accuracy sign**: GSM8K is -0.4 pt in all 4 runs. Check whether it comes from the MXFP8 **re-quantization** of the weights: compare per-layer weight error (block-128 dequant vs MXFP8 dequant) and, if cheap, run base with the prototype forced to M ≥ 1 (all QuACK) vs forced off, greedy-diff on 64 prompts.
Deliverable: `results/worker/W24/SUMMARY.md`. Keep W19 as is (waiting on the owner).
