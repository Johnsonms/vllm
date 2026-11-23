# STATUS_WORKER (worker-owned)

Dated log, newest entry first. Times are UTC.

## Current
- **W0: done** (2026-11-23). Node is 8x B200, so W1 applies.
- **W1: ack, waiting for an idle node.** A different workload on this node is using GPUs 0 and 1 (short repeated jobs, not started by the worker). Per rule 6 no timed run starts until all 8 GPUs are idle. The worker polls `nvidia-smi` and starts W1 as soon as they are.

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
