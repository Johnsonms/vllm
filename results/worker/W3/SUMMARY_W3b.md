# W3b — PR #62300 test, sleep test, skip fix, hang location

Worker node, GPUs 4-7 (idle before each run), vLLM 8399da3780 + local edits. All edits were local and are reverted;
`git status` in the vLLM tree is clean again. Same command as W3 (TEP4 + MTP3, dummy weights).

## Results

| # | Question | Answer |
|---|---|---|
| 1 | Does #62300 remove the truncation? | **No.** It failed on the 2nd case. |
| 2 | Does #62300 + a 50 ms sleep before the profiler stops? | **Yes.** 10 of 10 cases complete on all ranks. |
| 3 | Skip fix | Splitting replays by `cudaGraphLaunch` correlation works; the rendezvous all-reduce leaves the forward profile. |
| 4 | Where do ranks block after raising? | In `torch.distributed.destroy_process_group`, called from the engine's exit-time shutdown, on every rank. |

## 1. PR #62300 alone — does not fix it

Only the PR's `vllm/benchmarks/forward/profile.py` hunk was applied. Its test-file hunk does not apply on 8399da3780.
The trace confirms the change is active: 0 `cudaProfilerStop` events (1 without the PR).

| case | records per rank | replays equal | last GPU record after last sync (us) | margin to window end (us) |
|---|---|---|---|---|
| d129_1 | 49,341 on all 4 | yes | -62 … +17 | 288 … 452 |
| f192_1 | **70,505 on all 4** (full = 70,509) | **no** | +267 … +325 | **13 … 73** |

All four ranks raised `70505 GPU activities don't split into 3 replays`, and the run then hung to the timeout. The margin
is the same few hundred microseconds as without the PR, so nested `cudaProfilerStop` is not the cause.

**Correction to my own process:** two earlier runs meant for steps 1 and 2 ran unmodified code, because `git apply` of the
full PR diff failed on the test file and I did not check. Both failed at f192 (1 of 4 and 3 of 4 ranks truncated). They
count only as extra baseline failures. The numbers above are from the rerun with the diff verified.

## 2. PR #62300 + 50 ms sleep — fixes it

Local edit, inside the `torch.profiler.profile` block of `profile_graph`:

```python
with torch.profiler.profile(activities=activities) as prof:
    run(mark_for_nsys=False)
    time.sleep(0.05)   # keep the record window open past the last replay
```

5 x `129*1@4096` and 5 x `192*1@4096`, exit code 0:

| cases | records per rank | replays equal | last GPU record after last sync (us) | margin to window end (us) |
|---|---|---|---|---|
| d129_1..5 | 49,341 on all 4 ranks, all 5 cases | yes | -85 … +58 | 50,322 … 50,640 |
| f192_1..5 | 70,509 on all 4 ranks, all 5 cases | yes | -14 … +554 | 49,890 … 50,478 |

- GPU records still end up to 0.55 ms after the host's last `cudaDeviceSynchronize` returned. The sleep does not change
  that; it only moves the window end out of reach.
- Without the sleep, f192 failed in 4 of 4 runs where it was reached (baseline x3 with the two invalid runs, PR x1).
  So at 192 requests the failure is close to systematic on this node, not rare.
- The sleep was tested only together with the PR, not on its own.

## 3. Skip fix

**Where the 2 extra activities come from.** They are not in the graph. Before every `cudaGraphLaunch` the host launches two
eager `vectorized_elementwise_kernel<2, FillFunctor<long>>` kernels through `cudaLaunchKernel` (6 per trace = 2 x 3
replays). With `--exclude-sampler` and no `--speculative-config` there are none. This fits PyTorch refreshing a CUDA RNG
generator's seed and offset tensors before replaying a graph that uses it; I did not confirm that in the PyTorch source,
and I did not separate sampler from drafter. Nothing in `capture.py` launches them directly.

**Fix tested** (local edit in `summarize()`): keep only activities whose correlation ID belongs to a `cudaGraphLaunch`
runtime event, group them per launch, and raise if the per-replay counts differ. `skip = len(state_views) + 1` is then
exact and needs no change.

```python
launches = [e["args"]["correlation"] for e in events_by_ts
            if e.get("cat") == "cuda_runtime" and e.get("name") == "cudaGraphLaunch"]
by = {c: [a for a in acts if a["args"].get("correlation") == c] for c in launches}
if len(launches) != replays or len({len(v) for v in by.values()}) != 1:
    raise ValueError(f"per-replay GPU activity counts differ: {[len(v) for v in by.values()]}")
```

Offline, on the stored W3 traces (rank 0, old summary -> new summary):

| case | activities | busy_us | span_us | rendezvous f32 all-reduce in forward (us) | event-timed median (us) |
|---|---|---|---|---|---|
| a129 | 1,991 -> 1,989 | 27,018 -> 19,152 | 23,634 -> 15,765 | 7,695 -> gone | 15,616 |
| c128 | 2,042 -> 2,040 | 19,616 -> 19,177 | 16,103 -> 15,647 | 1,851 -> gone | 15,516 |
| e160 | 1,991 -> 1,989 | 21,550 -> 20,948 | 18,046 -> 17,425 | 456 -> gone | 17,268 |
| g256 | 1,990 -> 1,988 | 24,965 -> 23,134 | 21,669 -> 19,865 | 1,978 -> gone | 19,676 |

- `ncclDevKernel_AllReduce_Sum_f32_RING_LL` no longer appears in the forward kernel list.
- `busy_us` drops by the rendezvous time plus one restore copy (about 1.5 us), as expected.
- `span_us` now lands within about 1 % of the event-timed median; before, it was off by up to 50 %.
- The same check turns both known truncated traces into explicit errors, including the one that had passed silently:
  `f192.rank0 -> per-replay GPU activity counts differ: [23501, 23501, 23495]` and
  `d129.rank3 -> [16445, 16445, 15631]`.

Live, with the fix (plus PR and sleep): d128 gives 2,040 activities, span 15,659 us against a median of 15,512 us;
f192 gives 1,989 activities, span 18,265 us against 18,063 us; no f32 all-reduce in either `kernels.json`.
The existing `summarize` unit test still passes (the fix falls back to the old split when a trace has no
`cudaGraphLaunch` events).

## 4. Hang after an exception

Method: local debug hook `faulthandler.dump_traceback_later(120, repeat=True)` per rank, and a forced `ValueError` in
`profile_graph` on all ranks for one case. All four ranks printed the traceback, none exited, and the run hit the timeout.
The stack dumps are identical on all four ranks and unchanged over 4 dumps (8 minutes):

- **Main thread:** interpreter exit -> weakref finalizer -> `EngineCore.shutdown` (`vllm/v1/engine/core.py:898`) ->
  `cleanup_dist_env_and_memory` -> `destroy_distributed_environment` -> `GroupCoordinator.destroy`
  (`parallel_state.py:1427`) -> `torch.distributed.destroy_process_group` (`distributed_c10d.py:3060`). Blocked there.
- **A second thread:** `pynccl.py:269 _abort` -> `pynccl_wrapper.py:634 ncclCommAbort`. Also blocked.

So the hang is in shutdown, not in a collective of the next case, and it happens even when every rank fails at the same
point. This corrects my W3 guess that peers wait for the failed rank. Why `destroy_process_group` blocks was not
investigated; a clean run exits normally through the same benchmark, so the difference is the exception path
(`run_case` does not get to release the benchmark's graph and state before the exit-time shutdown runs).

## Not done

- The 50 ms sleep without the PR.
- Whether the late GPU records are real execution after the sync or a timestamp offset. The sleep run shows the records
  are consistently late at 192 requests (+100 … +554 us) and not at 129 (-85 … +58 us), which looks more like real
  work finishing after the host sync than like random clock skew, but that is still an inference.
- Root cause of the `destroy_process_group` block.
