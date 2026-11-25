# W13 — shared-expert aux-stream priority and order on B200 (interim, 03:47Z)

Qwen3.6-35B-A3B-FP8, TP1, real weights, B3 serve flags, compile cache off (0 AOT loads in every server). Base = main at 04f0149d47. **This base does not include #69736's SM100 skinny table**, so both arms run the cuBLAS BF16 GEMMs (different from B7, where the skinny plans are on in both arms).

Status: **P (aux-stream priority) measured: -6.7 % TPOT at 4 requests, -1 to -2 % at 1 and 2, nothing at 8 and 16; output identical.** O is already what main does. One optional arm (P stacked on the W12c fusion) is running.

## Result
| requests | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| base TPOT p50 (ms), pass 1 / pass 2 | 3.541 / 3.522 | 3.732 / 3.721 | 4.459 / 4.434 | 5.128 / 5.125 | 6.074 / 6.055 |
| P TPOT p50 (ms), pass 1 / pass 2 | 3.480 / 3.492 | 3.690 / 3.700 | 4.119 / 4.122 | 5.089 / 5.094 | 6.029 / 6.031 |
| P vs base, pass 1 (base GPU 4, P GPU 5) | -1.36 % | -1.68 % | **-6.91 %** | -0.62 % | -0.67 % |
| P vs base, pass 2 (GPUs swapped) | -2.18 % | -0.90 % | **-6.39 %** | +0.04 % | -0.37 % |
| **mean of the two passes** | **-1.8 %** | **-1.3 %** | **-6.7 %** | **-0.3 %** | **-0.5 %** |
| pairs with P faster | 11 / 12 | 12 / 12 | 12 / 12 | 9 / 12 | 9 / 12 |

- Method: both servers up at once, 6 rounds per concurrency, 5 x c prompts per burst, 512/512 random, order flipped each round, then GPUs swapped. Percentages are the mean of the 6 per-pair differences; the ms rows are medians of the 6 bursts.
- At 1 request, pass 2 has one outlier pair (-10.2 %). From the medians the gain at 1 request is about -1.3 %.
- At 4 requests the 12 pairs range from -5.0 % to -7.9 %.
- Greedy output: 4 prompts, 256 tokens each, P identical to base byte for byte.
- GSM8K, 2 runs per arm (1319 questions, 64 concurrent): base 0.9629 / 0.9636, P 0.9591 / 0.9591. The difference is 5-6 questions. Single-request output is identical, so I read this as batch-composition effects under 64 concurrent requests (base differs from itself by one question). I did not check it per question.
- **Node was not idle for part of this**: another job of the owner started on GPUs 0 and 1 during the second pass. My runs were on GPUs 4 and 5. Both arms saw it at the same time.

## 1. Does stream priority survive CUDA-graph replay? Yes.
- The edit (P): `aux_stream()` creates its stream with priority -1 instead of the default 0. One argument.
- Device priority range on B200: least 0, greatest -5.
- torch maps `torch.cuda.Stream(priority=-1)` to a CUDA stream of priority -1 (checked with `cudaStreamGetPriority`).
- A captured graph keeps it per node. `prio_probe.py` captures a small two-stream graph with `keep_graph=True` and reads `cudaGraphKernelNodeGetAttribute(node, cudaLaunchAttributePriority)`: the aux-stream kernel nodes report -1 (or -2 when asked), the main-stream nodes 0.
- torch 2.14 instantiates with `cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority` (`aten/src/ATen/cuda/CUDAGraph.cpp`, same on torch main). So no torch change is needed.
- It also acts in the server: in the profile the shared expert's `act_and_mul` start moves (table below).

## 2. Order (O): already done on main
- `Qwen3NextSparseMoeBlock` passes its router (`gate`) to the MoE runner. In the runner's `_forward_impl`, `maybe_forward_async` (the aux-stream enqueue of the shared expert) is called first; the router GEMM runs after it. So the shared expert is already enqueued before the router GEMM. No prototype.
- In the trace this is visible: the shared expert's quant and gate-up GEMM start before the router's split-K reduce ends.
- I tried one different order of my own (arm G): inside the shared-expert MLP, compute the (1, 2048) gate GEMV first instead of last. Profile only, no A/B: it delays `act_and_mul` (the GEMV takes the early slot) and the layer does not get shorter. Not pursued.

## 3. What P does to a layer (profile, B6 method, median over 1248 layers; us relative to the routed gate-up GEMM start)
| requests | arm | `act_and_mul` start | shared branch end | routed branch end (finalize) | routed gate-up GEMM | routed down GEMM |
|---|---|---|---|---|---|---|
| 1 | base | +3.6 | 26.0 | 16.5 | 10.2 | 6.9 |
| 1 | P | +4.6 | 26.3 | 16.1 | 9.3 | 6.4 |
| 4 | base | +11.4 | **39.8** | 34.1 | 18.8 | 15.5 |
| 4 | P | +5.8 | 31.6 | **33.9** | 18.4 | 15.1 |
| 8 | base | +17.4 | **55.6** | 43.5 | 27.0 | 18.7 |
| 8 | P | +2.8 | 45.8 | **51.8** | 30.2 | 21.8 |

Bold = the later branch, which sets the end of the layer.
- **4 requests:** in base the shared branch ends 5.6 us after the routed one. With P it ends 2.3 us before it, and the routed GEMMs are not slower. The layer ends about 6 us earlier (x 40 layers = 240 us, the size of the measured gain).
- **8 requests:** P moves `act_and_mul` from +17 to +3 us, but the routed GEMMs get slower by the same amount (gate-up 27.0 -> 30.2, down 18.7 -> 21.8). The layer end moves from 55.6 to 51.8 us in the profile, but the A/B shows no gain. At 8 requests the two branches together fill the SMs, so priority only changes who waits.
- **1 request:** the timeline is unchanged by P (the shared branch is not waiting; it is just longer). The small A/B gain at 1 and 2 requests is not explained by this table.
- Priority -5 gives the same timeline as -1.
- The profiles of the five arms ran at the same time on five GPUs, one run each, so I do not compare their absolute step times; only the within-layer offsets.
- Timelines of one layer: `layer_base_c4.txt`, `layer_p1_c4.txt`, `layer_base_c8.txt`, `layer_p1_c8.txt` (B300's `layer_view.py`).

## Open
- Optional arm running: W12c fusion (my local prototype, not the #69778 code) with and without P, same paired method. It tells whether the two gains add.
- P changes a process-wide stream. The comment in the source says the aux stream is used only for the shared-expert overlap; I did not test other models.

## Files
`prio_probe.py` (priority in graph nodes), `prof.sh`, `layers.py` (table in section 3), `paired.sh`, `pairsum.py`, `tokens.sh`, `gsm.sh`, `layer_*.txt`.
