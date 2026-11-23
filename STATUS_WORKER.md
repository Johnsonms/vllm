# STATUS_WORKER (worker-owned)

Dated log, newest entry first. Times are UTC.

## Current
- **2026-11-23 23:55Z — W3b: done.** See `results/worker/W3/SUMMARY_W3b.md`. (1) PR #62300 alone does not fix the truncation: f192 lost 4 records on all 4 ranks. (2) PR + 50 ms sleep before the profiler stops: 10 of 10 cases complete, margin ~50 ms. (3) Splitting replays by `cudaGraphLaunch` correlation removes the rendezvous all-reduce from the forward profile, brings `span_us` within ~1 % of the timed median, and turns truncated traces into explicit errors; the 2 extra kernels are eager RNG-state fills launched before each replay. (4) After an exception every rank blocks at exit in `torch.distributed.destroy_process_group`. Two of my earlier runs were invalid (the PR diff had not applied); they are disclosed in the summary and not used. Local vLLM edits are reverted.
- **2026-11-23 22:20Z — W3: done.** See `results/worker/W3/SUMMARY.md`. Reproduced once (129 tokens, rank 3, other ranks hung). No kernel explodes: activities per forward stay near 2,000 and the 16.4k figure includes the linear KV-restore copies. The error comes from GPU records stamped after the profiler's window end being dropped from the last replay on one rank; it is flaky, and one trace lost 6 records and passed silently. Hang cause not confirmed by stack dump. One accounting bug found (rendezvous all-reduce counted as a forward kernel). A 50 ms sleep before the profiler stops would test the theory but needs a code edit, so it waits for your OK.
- **W2: still parked.** At 22:20Z GPU 0 holds ~61 GB from the other workload; GPUs 1-7 are free.
- **2026-11-23 21:10Z — W2 (revised, serving): ack, waiting for an idle node.** Read all manager notes up to the 21:15Z STOP note. W1b and W1 are skipped as instructed; nothing was run for them, so there are no partial results.
- **Worker missed the 19:45Z go-ahead.** The worker's idle check required zero GPU processes and it did not fetch the branch while waiting, so the window with one idle process on GPU 1 was not used. Sorry. From now on the worker fetches every 10 min while waiting.
- **The node is now actively shared, not just holding an idle process.** At 21:03Z the other workload holds ~23 GB on GPU 0 and ~58 GB on GPU 1 and starts new jobs every few minutes. This is outside the go-ahead (one idle process, 0 % utilisation), and TP8 serving at `--gpu-memory-utilization 0.85` would not fit next to 58 GB on GPU 1 anyway. The worker has not touched that workload.
- **W2 script ready:** `results/worker/W2/w2_serve.sh`, copied from `m2b_serve.sh` with model Qwen/Qwen3-32B, no expert parallel, arms default / 16 / 64, and `VLLM_LOGGING_LEVEL=DEBUG` in every arm so each `serve.log` shows the `Flashinfer Allreduce Fusion: <max_token_num>` line. Expected limits at hidden 5120: default 1 MB = 102 tokens, 16 MB = 1638, 64 MB = 6553. It starts as soon as no GPU shows utilisation and GPUs 0/1 hold less than 2 GB.
- **W2-prep: done.** Qwen/Qwen3-32B is in the HF cache (62 GB, 17 shards).
- **W1-prep: done** (TP4 functional run on GPUs 4-7, all fused and standard row families present; timings discarded).
- **Same MNNVL warning here:** "FABRIC handles failed on this host; using POSIX file descriptors instead", `CUDA_ERROR_NOT_PERMITTED` on every rank (no IMEX channel).

## W0 report

| Item | Worker node | Manager node (SETUP.md) |
|---|---|---|
| GPUs | 8x NVIDIA B200 | 8x B200 |
| Driver | **580.159.04** | 595.91.07 |
| System CUDA toolkit | **13.1** (nvcc 13.1.115) | 13.2 |
| torch | 2.14.0+cu132 | 2.14.0+cu132 |
| vLLM | 0.34.0rc1.dev1161+g8399da378 (commit 8399da3780, precompiled) | same |
| FlashInfer | 0.7.4 | 0.7.4 |
| Triton | 3.8.0 | 3.8.0 |
| Python | 3.12 (uv venv) | 3.12 |
| `vllm download-kernels` | succeeded: `flashinfer-jit-cache==0.7.4+cu130`, including `sm100a` | — |

Differences from the manager: the driver (580 vs 595) and the system CUDA toolkit (13.1 vs 13.2). torch ships its own cu132 runtime, so the Python stack is identical. Keep the driver difference in mind if W1 numbers disagree with the manager's cross-check.

`nvidia-smi -L`: GPU 0–7, all `NVIDIA B200`.

`nvidia-smi topo -m` (GPU block): every GPU pair is `NV18`. GPUs 0–3 are on NUMA node 0 (CPUs 0–79), GPUs 4–7 on NUMA node 1 (CPUs 80–159).

```
      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7  CPU Affinity  NUMA
GPU0   X   NV18 NV18 NV18 NV18 NV18 NV18 NV18  0-79          0
GPU1  NV18  X   NV18 NV18 NV18 NV18 NV18 NV18  0-79          0
GPU2  NV18 NV18  X   NV18 NV18 NV18 NV18 NV18  0-79          0
GPU3  NV18 NV18 NV18  X   NV18 NV18 NV18 NV18  0-79          0
GPU4  NV18 NV18 NV18 NV18  X   NV18 NV18 NV18  80-159        1
GPU5  NV18 NV18 NV18 NV18 NV18  X   NV18 NV18  80-159        1
GPU6  NV18 NV18 NV18 NV18 NV18 NV18  X   NV18  80-159        1
GPU7  NV18 NV18 NV18 NV18 NV18 NV18 NV18  X    80-159        1
```

## Questions
- (none)
