# STATUS_WORKER (worker-owned)

Dated log, newest entry first. Times are UTC.

## Current
- **W0: done** (2026-11-23). Node is 8x B200, so W1 applies.
- **W1: ack, still waiting for an idle node.** One process from the other workload still holds GPU 1 (0 % utilisation, ~0.9 GB). The worker has not touched it. Idle check continues every ~10 min; `blocked` will be posted if the node is still shared 2 h after the manager's reply.
- **W1-prep: done.** The W1 command ran end to end at TP4 on GPUs 4-7 (hidden 5120, tokens 128 and 1024, 5 trials, exit code 0). The output table has all row families: Flashinfer Mnnvl fused (oneshot, twoshot), Flashinfer Trtllm fused (oneshot, twoshot, and with FP8 quant), and the Standard Allreduce rows (custom/native RMSNorm, custom/native FP8 quant, native compiled). Timings are discarded as agreed.
- **Same MNNVL warning here:** "FABRIC handles failed on this host; using POSIX file descriptors instead", `CUDA_ERROR_NOT_PERMITTED` on every rank (no IMEX channel).
- **W2-prep: running.** Qwen/Qwen3-32B download into the HF cache is in progress.
- Read the W1 methodology note. The benchmark output does not name the standard path's backend, so the worker will run `results/manager/ar_probe.py` at TP8 alongside W1 once the node is idle and report the backend per size.

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
