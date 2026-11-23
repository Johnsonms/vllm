# W3 — `vllm bench forward --profile-dir` fails with "GPU activities don't split into 3 replays"

Worker node: 8x B200, driver 580.159.04, vLLM 8399da3780 unmodified, GPUs 4-7, TEP4 + MTP3, dummy weights.
GPUs 4-7 were idle for every run. No vLLM code was changed.

## Short answer

1. **No kernel explodes. This is not a vLLM perf bug.** Activities per forward stay at about 2,000 from 48 to 256 tokens.
2. **The count is not divisible by 3 because the trace loses the tail of the last replay on one rank.** GPU records whose
   timestamps fall after the profiler's "Record Window End" are dropped. The margin between the last GPU record and the
   window end is only a few hundred microseconds, and it is sometimes used up.
3. **It is flaky, not a 128-token threshold.** The same command passed at 128 here and failed at 128 on the manager node;
   129 failed here once and passed three times.
4. **It can also pass silently with a wrong split**: one trace lost exactly 6 records, so the count still divided by 3.

## 1. Which activities "explode" — none

The "~2k at d96 vs ~16.4k at d128" comparison mixes two different numbers. 2k is the printed `activities/forward`, which
excludes the pre-forward part. 16.4k is the error message's total per replay, which includes it.

| case | total per replay | pre-forward (restore copies + rendezvous) | activities/forward |
|---|---|---|---|
| d48  | 7,222  | 5,384  | 1,838 |
| d96  | 12,802 | 10,760 | 2,042 |
| d127 | 16,223 | 14,232 | 1,991 |
| d128 | 16,386 | 14,344 | 2,042 |
| d129 | 16,447 | 14,456 | 1,991 |
| d160 | 19,919 | —      | 1,991 |
| d192 | 23,503 | —      | 1,989 |
| d256 | 30,670 | —      | 1,990 |

The pre-forward part is the benchmark restoring its KV state views before each replay, one small copy per view, about 112
per request. It grows linearly with the request count and is not part of the timed forward. (Totals for d160-d256 are
count/3 from the repeat run; their pre-forward split was not extracted.)

One real change happens between 96 and 127 tokens, and it is harmless: about 13 per request of those restore copies are
reported by CUPTI as `gpu_memcpy` "Memcpy DtoD" records instead of `memcpy32_post` kernel records (1,255 kernel records at
d96; 1,671 memcpy records at d128). Same copies, different record type.

## 2. Why the count is not divisible by 3

Failing run here: case `129*1@4096`, rank 3, `ValueError: 48527 GPU activities don't split into 3 replays`.
Ranks 0-2 of the same case each have 49,341 (= 3 x 16,447).

- Rank 3 is missing **814** records. The loss is spread over about 60 kernel names, which rules out one misbehaving kernel.
- All 814 are the **end of the third replay**: after that replay's rendezvous all-reduce, rank 3 has 1,176 records and the
  other ranks have 1,990. Replays 1 and 2 are complete (16,447 each).
- Rank 3's last kept record ends **17 us before** the trace's `Record Window End` marker.
- That last record ends 354 us *after* the host's final `cudaDeviceSynchronize` returned, and the window closes 372 us
  after that sync.

The manager's failing count fits the same pattern: 49,142 = 49,158 - 16, where 49,158 is this node's d128 total.

Repeat run (7 cases, same command, all passed the check). "after sync" is how far the last GPU record ends after the
last host sync returned; "margin" is what is left before the window end:

| case.rank | records | after sync (us) | margin (us) |
|---|---|---|---|
| a129 (4 ranks) | 49,341 | -53 … -14 | 359 … 460 |
| c128 (4 ranks) | 49,158 | -3 … +26 | 345 … 399 |
| e160 (4 ranks) | 59,757 | -17 … +127 | 253 … 373 |
| f192.rank0 | **70,503** | +445 | 65 |
| f192.rank1 | 70,509 | +396 | 106 |
| f192.rank2 | 70,509 | +264 | 247 |
| f192.rank3 | 70,509 | +483 | **17** |
| g256 (4 ranks) | 92,010 | -14 … +382 | 791 … 1,256 |

- GPU records end up to ~0.5 ms after the host sync returned, while the window closes only ~0.3-0.5 ms after that sync.
- **f192.rank0 lost 6 records and still passed**, because 70,503 is divisible by 3. Its three "replays" are then cut at
  the wrong places, so its `kernels.json` is slightly wrong without any error.

**Not settled:** whether the GPU work really ends after the sync returns, or whether CUPTI's GPU timestamps are offset from
the host clock by a few hundred microseconds. Either way the cause of the error is the same: records stamped after the
window end are dropped. In the failing run the ranks also disagreed by milliseconds (rank 0: -6.6 ms, rank 3: at least
+0.35 ms), which points to timestamp conversion rather than real execution, but this is a guess.

**Proposed test (needs a code edit, so not done):** in `vllm/benchmarks/forward/profile.py::profile_graph`, sleep ~50 ms
after `run()` and before leaving the `torch.profiler.profile` block. If the theory is right, all ranks keep full counts.
A sturdier fix for `summarize()` is to split replays at the rendezvous all-reduce instead of dividing the count by 3, and
to compare counts across replays.

## 3. Why the other ranks hang

- `summarize()` is local to each rank, so only the rank that lost records raises; the others get a clean summary.
- Rank 3 printed its full traceback (through `run_case` and `main`) but its process did not exit: torchrun only stopped
  the workers when the outer `timeout` sent SIGTERM 30 minutes later.
- So rank 3 hangs during shutdown, and ranks 0-2 have meanwhile moved on to the next case and wait in a collective for it.
  Note that `run_case`'s `finally` runs `runner.execute_model(batch.finish())` on rank 3, a forward with collectives that
  the peers also run once, so the ranks stay paired until rank 3 leaves the case loop.
- **Not confirmed with a stack dump.** `ptrace_scope` is 1 on this node, so py-spy cannot attach to the ranks, and the
  failure did not recur in the repeat run.

## Side finding: the rendezvous all-reduce is counted as a forward kernel

`skip = len(state_views) + 1` is two short of the real pre-forward activity count: each replay starts with two extra
fill kernels (`vectorized_elementwise_kernel<2, FillFunctor...>`). The last skipped slot therefore lands one before the
rendezvous, and `ncclDevKernel_AllReduce_Sum_f32_RING_LL` (the 1-element flag all-reduce) is reported inside the forward.
It only waits for the slowest rank, so its time is noise: 8 us in one replay, 5,220 us in another. In the d128 summary it
is the #2 "kernel" at 10.4 % and inflates `busy_us`. The timed latency uses in-graph events and is not affected.

## Minimal repro

The manager's command with `--case d129=129*1@4096` (one case is enough; repeat it a few times, it fails about 1 in 4-10
cases on this node). Check any trace directory with `results/worker/W3/trace_tail.py <dir>`: unequal record counts across
the ranks of one case, with a margin near 0, mark a truncated trace.

Variants without `--speculative-config` and with `--routing zipf` were **not run**: the failure is in the profiler window,
not in the model path, so they would not separate anything. Say if you still want them.

Raw traces (about 1.2-2 MB per rank and case, 60 files) are kept on the worker node, not in this branch.
