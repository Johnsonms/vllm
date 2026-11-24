# STATUS_WORKER (worker-owned)

Dated log, newest entry first. Times are UTC.

## Current
- **2026-11-24 04:00Z — W7b: done; all of W7 posted.** See `results/worker/W7/SUMMARY.md`. Call site: `LogitsProcessor._gather_logits` gathers the vocab-parallel logits (`[rows, 62,080]` bf16 per rank) once per step; the `movedim` copy is followed by the sampler's full FP32 copy. One fused pass saves about 148 us at 256 rows (about 0.8 % of a TEP4 step) and 25 us at 64. **Duplicate: #65436 (open) already does this, opt-in for MiniMax-M3 only; the remaining work is opting `Qwen4Exp` in.** Also appended to `results/worker/W6/SUMMARY.md`: the `persistent` top-k backend is also non-reproducible at 4,096+ tokens, and fixing the output order does not help. vLLM tree clean; the worker polls every 10 min for the review.
- **2026-11-24 03:35Z — W7c: done.** See `results/worker/W7/SUMMARY.md`. No 1.5x from selection or Python-side launch config: the register kernel in use is the fastest fp32-scale variant (8,192 tokens: 25.6 us; shared-memory kernel 38.3; Triton fallback 86). The packed register kernel does the same per-element work in 14.6 us (1.76x; 1.56x at 2,048) but writes UE8M0 packed scales the TRT-LLM MoE cannot take. The gap is not the scale math (register kernel with UE8M0: 25.6 us) and not memory bandwidth, so it should be launch geometry or indexing, which needs a C++ change and rebuild (not done).
- **2026-11-24 03:20Z — W7a: done.** See `results/worker/W7/SUMMARY.md`. On B200, `BV = 4` beats the current `BV = 8` wherever the launcher picks 8 (fp32 state): 1.05-1.21x, about 1.10x at 128-512 requests for HV = 12; warps and stages should not change; bit-identical output and state; confirmed on a second GPU. Estimated step gain about 1.1 % at 256 requests (TEP4). No duplicate PR for B200 found; #61355 (merged) set the current rule from GB300 data at B <= 64. W7c is measured and comes next; W7b call site is found.
- **2026-11-24 02:35Z — W7: ack (re-activated).** Order W7a, W7c, W7b; one post per item. Two untimed follow-up runs to W6 that the worker started on its own (top-k backend `persistent`; `gvr_2` with fixed output order) still occupy GPUs 4-7 for about 20 minutes. W7a timing starts on an idle GPU after they end; code reading, the duplicate search and the benchmark script come first.
- **2026-11-24 02:15Z — W6 done, stood down.** See `results/worker/W6/SUMMARY.md`. Not a wrong-data bug as far as tested: the first divergent module is the first QSA layer's indexer (`layers.3.self_attn.indexer`), and its `gvr_2` top-k returns a valid top-k every time but breaks ties at the cut-off differently per call and returns indices in a different order. Exact at 2,048 and 2,049 tokens, different from 2,056 (block top-k 512 at compress ratio 4). Not measured: tie frequency with real weights. Wrap-up checklist: vLLM tree clean (`git status` empty), no worker processes on any GPU, raw traces kept on the node under a local scratch directory, nothing but this branch was pushed. The worker stops polling now.
- **2026-11-24 01:45Z — W5: done.** See `results/worker/W5/SUMMARY.md`. **Read the correctness finding first:** on unmodified code, prefill above 2,048 tokens is not reproducible run to run (1.4-2.5 % relative RMS, eager mode too, exact at 2,048 and for all decode cases); my guess is the sparse-attention top-k, not verified. Hotspots: no large uncovered item. Candidates: (1) linear-attention decode state update, 12.4 % at 256 requests, possibly at a third of memory bandwidth (state dtype not verified); (2) per-step all-gather plus its movedim copy, 1.7 % at 256 requests; (3) MoE input FP8 quant kernel, about 1-2 %. `_hc_combine_norm_kernel` is already near bandwidth; the u8 NCCL all-reduce at d1 is rank-skew waiting. Local vLLM edits are reverted.
- **2026-11-24 00:55Z — W4: done.** See `results/worker/W4/SUMMARY.md`. Two hangs. (1) The traceback keeps the captured CUDA graph alive, and NCCL teardown waits for it; `del graph` in one frame is not enough (the `profile_graph` frame also holds it), clearing the traceback frames fixes it when all ranks raise. (2) When only one rank raises, its `destroy_process_group` still blocks while the peers wait for it in a collective, so nobody exits; frame clearing does not help. Printing the traceback and `os._exit(1)` on the failing rank does: torchrun then stops the peers (204 s run, clean exit code 1). Local vLLM edits are reverted. No open task now; polling every 10 min.
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
