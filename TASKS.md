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
