# H2-0: bring-up of the second Hopper node (hopper2)

## Node
- 8x NVIDIA H100 80GB HBM3 (SM90, capability (9, 0)), VBIOS 96.00.D0.00.02, PCIe Gen5 x16 each.
- Driver 610.57.04 (CUDA 13.3 UMD); system CUDA toolkit 13.3 present (`nvcc` available).
- Topology (`topo.txt`): every GPU pair NV18 (full NVLink mesh). GPUs 0-3 → NUMA 0 (CPUs 0-63), GPUs 4-7 → NUMA 1 (CPUs 64-127). 128 CPUs, ~885 GB RAM.
- `numactl` not installed yet; will pin with `taskset` (like the first node) unless installed.

## Software (`versions.txt`), base vLLM `cfbc42099a`, env per `setup/HOPPER_START.md` §2
python 3.12.16, vllm 0.34.1rc1.dev321+gcfbc42099.precompiled, torch 2.14.0+cu132 (CUDA 13.2), Triton 3.8.0, FlashInfer 0.7.4, **quack-kernels 0.6.5**, nvidia-cutlass-dsl 4.8.0, cupti-python 13.2.0, transformers 5.24.1. Same stack as the first Hopper node.

## GPU health
Idle (`idle_gpus.csv`): all 8 GPUs 28-33 °C, 70-73 W, P0, 0 uncorrected ECC errors. **No hot outlier** (first node's GPU 0 idled at ~71 °C).

30 s BF16 GEMM, 8192³, all 8 GPUs at once (`gemm_health.py`, `gemm_gpu*.json`, `under_load.csv`):

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| TFLOPS (30 s mean) | 682.2 | 695.3 | 679.3 | 683.1 | 683.6 | 688.0 | 685.9 | 683.0 |
| Temp under load (°C) | 56 | 56 | 62 | 58 | 57 | 59 | 56 | 53 |

- Spread 679-695 TFLOPS (2.3%, max/min); every GPU sits at the 700 W cap (event reason 0x4 = SW power cap, no thermal flags) at 1365-1380 MHz.
- **All 8 GPUs healthy; none excluded.** GPU 2 runs a few °C warmer under load but is not slower.

## Next
H2-4 (QuACK `gemm_w4` vs vLLM int4 on Qwen3-8B/32B-AWQ), starting now. Note from the source: on SM90 the mixed-precision order is CutlassW4A8 → Machete → Humming → Marlin (Humming is moved ahead of Marlin on SM90), so AWQ is expected on Machete; will confirm from serve logs.
