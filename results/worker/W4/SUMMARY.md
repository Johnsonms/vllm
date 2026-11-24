# W4 — why `vllm bench forward` hangs at exit after an exception

Worker node, GPUs 4-7 (idle before each run), vLLM 8399da3780 + local debug edits, TEP4 + MTP3, dummy weights.
All edits were local and are reverted; the vLLM tree is clean. Failures were forced with a debug `raise` in
`profile_graph`; stacks come from `faulthandler.dump_traceback_later`.

## Short answer

There are **two** hangs, and the minimal change fixes only the first.

1. **Own graph kept alive.** The exception's traceback keeps the captured CUDA graph alive until interpreter exit. NCCL
   teardown waits for that graph, so the rank blocks in `destroy_process_group`. Clearing the traceback's frames fixes it.
2. **Cross-rank deadlock.** When only one rank raises (the real "don't split into 3 replays" case), that rank's
   `destroy_process_group` still blocks, because the peers are alive and waiting for it in a collective. The failed rank
   never exits, so torchrun never stops the peers. Only exiting the failed rank without the collective teardown fixes it.

## Experiments

| # | Who raises, and where | Local change | Result |
|---|---|---|---|
| 0 | all 4 ranks, before the replays (W3b run) | none | hang to timeout; all ranks in `destroy_process_group` |
| A | all 4 ranks, before the replays | `del graph` in `_capture_and_time` before re-raising | **hang**, same stacks |
| B | all 4 ranks, before the replays | clear traceback frames in `main()`'s case loop | **exits**: all ranks exit code 1 within seconds |
| B1 | rank 3 only, before the replays | same as B | **hang** |
| B2 | rank 3 only, after the replays (same point as the real error) | same as B | **hang** |
| C | rank 3 only, after the replays | fail fast: print traceback, then `os._exit(1)` | **exits**: 204 s total, rank 3 exit code 1, torchrun SIGTERMs ranks 0-2 |

Hanging runs were ended by a 720 s `timeout`. C was not run for the "all ranks raise" case; every rank would take the
same `os._exit(1)` path there.

## Hang 1: the traceback keeps the graph alive

- `pynccl.py::PyNcclCommunicator.destroy` already documents the rule: "`ncclCommAbort` can block until all CUDA graphs
  that captured NCCL ops on this comm are destroyed". The benchmark's graph captures collectives, including the
  rendezvous `torch.distributed.all_reduce` on `world.device_group`.
- On a clean run `_capture_and_time` ends with `del graph`. On an exception that line is skipped, and the traceback
  (kept by the interpreter until exit) references the frames whose locals hold the graph.
- **A fails because two frames hold it:** `_capture_and_time` and the callee `profile_graph(graph, ...)`, whose parameter
  is still alive in the traceback. Deleting one local is not enough.
- **B works** when all ranks raise:

```python
try:
    result = run_case(runner, case, args, sampling_params)
except BaseException as exc:
    traceback.clear_frames(exc.__traceback__)   # drops the graph, state snapshots and bench object
    gc.collect()
    raise
```

Stack of every rank in the hanging runs (unchanged across 4-6 dumps):
main thread `weakref finalizer -> EngineCore.shutdown (core.py:898) -> cleanup_dist_env_and_memory ->
destroy_distributed_environment -> GroupCoordinator.destroy (parallel_state.py:1427,
torch.distributed.destroy_process_group(self.device_group)) -> distributed_c10d.py:3060`;
a daemon thread in `pynccl.py:269 _abort -> ncclCommAbort`.

## Hang 2: one failed rank and live peers wait for each other

With B applied and only rank 3 raising:

| run | rank 3 (raised) | ranks 0-2 |
|---|---|---|
| B1, raise before the replays | `destroy_process_group` (same stack as above) | `torch.accelerator.synchronize` in `profile_graph`: their graph replay waits in the in-graph rendezvous for rank 3 |
| B2, raise after the replays | `destroy_process_group` | `torch.distributed.all_gather_object` on the CPU group in `main()`, waiting for rank 3's result |

- Rank 3's own graph is gone here (frames cleared), yet destroying the NCCL device group still blocks while the peers
  are alive and not tearing down. I did not look inside NCCL to see which call waits.
- Rank 3 never exits, so torchrun sees no failed child and does not stop ranks 0-2. That is the 30-minute hang from W3.

**C works** for this case. In the same `except` block, for multi-rank runs:

```python
traceback.print_exc()
sys.stderr.flush(); sys.stdout.flush()
os._exit(1)   # skip the collective teardown; the launcher then stops the peers
```

Result: the traceback is printed, rank 3 exits with code 1, torchrun sends SIGTERM to ranks 0-2 and reports
`ChildFailedError`. No GPU memory or processes were left on GPUs 4-7 afterwards.

## Recommendation

- Fail fast (C) on any exception in the case loop when `world_size > 1`. It covers both hangs.
- B alone is not enough for the real failure, where a single rank raises.
- `os._exit` skips all cleanup (atexit handlers, the usage-stats thread, temp files). That is acceptable for a benchmark
  that is already failing, but it is a blunt tool. A gentler variant was **not tested**: tell the peers first (for
  example abort the process groups on all ranks) and then exit normally.

## Not done

- Which NCCL call blocks in hang 2, and whether a newer NCCL changes it.
- C with all ranks raising, and C on a single-rank (TP1) run, where it is not needed.
- Whether the same exit hang affects `vllm serve` workers after an exception; this was only tested in `bench forward`.
