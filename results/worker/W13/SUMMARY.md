# W13 — shared-expert aux-stream priority and order on B200 (interim, 03:47Z)

Qwen3.6-35B-A3B-FP8, TP1, real weights, B3 serve flags, compile cache off (0 AOT loads in every server). Base = main at 04f0149d47. **This base does not include #69736's SM100 skinny table**, so both arms run the cuBLAS BF16 GEMMs (different from B7, where the skinny plans are on in both arms).

Status (superseded by the W13b section at the end): **P (aux-stream priority) measured: -6.7 % TPOT at 4 requests, -1 to -2 % at 1 and 2, nothing at 8 and 16; output identical.** O is already what main does. The optional stack arm and W13b are at the end.

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

# W13b (06:05Z): P with #69736 in both arms — still a gain on B200; GSM8K flips are noise

Tree: `sm100-skinny-qwen35-moe` at ce7e3d5dd7 (main 04f0149d47 + the SM100 table) in both arms; P = aux stream created with priority -1. Same paired method (6 rounds, 5 x c prompts, order flipped, GPUs 4/5 then swapped). Each server log shows the loaded file path and the priority in use. 0 AOT loads.

## 1. Paired A/B, c = 1 to 16
| requests | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| base TPOT p50 (ms), pass 1 / pass 2 | 3.401 / 3.378 | 3.709 / 3.687 | 4.179 / 4.172 | 4.942 / 4.937 | 6.090 / 6.100 |
| P TPOT p50 (ms), pass 1 / pass 2 | 3.306 / 3.327 | 3.555 / 3.568 | 3.951 / 3.960 | 4.876 / 4.901 | 6.060 / 6.065 |
| P vs base, pass 1 | -2.76 % | -3.94 % | -5.41 % | -1.32 % | -0.42 % |
| P vs base, pass 2 (GPUs swapped) | -1.44 % | -3.26 % | -5.03 % | -0.65 % | -0.46 % |
| **mean of the two passes** | **-2.1 %** | **-3.6 %** | **-5.2 %** | **-1.0 %** | **-0.4 %** |
| pairs with P faster | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | 10 / 12 |

- **No regression at any concurrency.** This is the opposite sign of B300's B7 at 4 requests (+2.5 % there).
- The skinny table itself moved the base: 4 requests 4.459 -> 4.179 ms, 8 requests 5.128 -> 4.942 ms (W13 base vs this base; not a paired comparison). P's gain at 4 requests went from -6.7 % to -5.2 %, and at 1-2 requests it grew (-1.8 / -1.3 % -> -2.1 / -3.6 %).
- Greedy output, 4 prompts, 256 tokens: identical byte for byte.
- c32 and c64 are running now with the same method. Result will be appended.
- Not idle: the owner's job is on GPUs 0-1, and my W14 profile runs on GPUs 2, 3, 6, 7 at the same time (the manager asked for W14 in parallel). Both arms see the same load.

## 2. GSM8K per question (tree with #69736, 3 runs per arm, 1319 questions, 64 concurrent, temperature 0)
`gsm_dump.py` runs the unchanged eval and also saves each question's prediction, label and text.

| run | base r1 | base r2 | base r3 | P r1 | P r2 | P r3 |
|---|---|---|---|---|---|---|
| correct | 1271 | 1271 | 1270 | 1270 | 1272 | 1267 |
| accuracy | 0.9636 | 0.9636 | 0.9629 | 0.9629 | 0.9644 | 0.9606 |

| compared runs | questions whose correctness differs | questions whose output text differs |
|---|---|---|
| base vs base (3 pairs) | 13, 15, 18 | 693-710 of 1319 |
| P vs P (3 pairs) | 16, 19, 19 | 696-722 |
| base vs P (9 pairs) | 11-18 | 652-725 |

- **Two runs of the same arm differ as much as two arms.** More than half of the outputs differ in text between any two runs, with base against itself too.
- 32 questions are not unanimous over the 6 runs. **None is right in all 3 base runs and wrong in all 3 P runs, and none the other way round.**
- So the 0.9629 / 0.9636 vs 0.9591 / 0.9591 in the interim was run-to-run noise. One run of this eval at 64 concurrent moves by about +-0.2 points (the 6 runs span 1267-1272 correct), and the earlier P runs were at the low end.
- Why outputs differ at all with temperature 0: at 1 request P is identical to base, so it is the batch composition under 64 concurrent requests (which requests share a step changes the batch size and the kernels picked). I did not isolate it further.
- For #69778: a change that is output-identical at 1 request sits inside a +-0.2 point band here, so a -0.4 point shift from 2 runs per arm is not evidence of a loss; 3 or more runs per arm and the per-question table are needed.

## 3. Optional stack arm (done before it was dropped): W12c fusion with and without P
Base 04f0149d47 **without** the skinny table, my local W12c prototype on in both arms. Paired, GPUs swapped.

| requests | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| fusion TPOT p50 (ms), pass 1 / pass 2 | 3.368 / 3.349 | 3.733 / 3.725 | 4.354 / 4.352 | 4.979 / 4.952 | 5.872 / 5.867 |
| fusion + P, pass 1 / pass 2 | 3.348 / 3.363 | 3.644 / 3.648 | 4.119 / 4.131 | 4.987 / 4.993 | 6.050 / 6.031 |
| **P on top of fusion, mean of passes** | **-0.1 %** | **-2.3 %** | **-5.2 %** | **+0.6 %** | **+3.0 %** |
| pairs with P faster | 6 / 12 | 12 / 12 | 12 / 12 | 2 / 12 | 0 / 12 |

- **With the fusion on, P costs 3.0 % at 16 requests (all 12 pairs).** Without the fusion P is neutral at 16. The fusion alone makes 16 requests faster (6.07 -> 5.87 ms across runs), and P gives that back.
- So P and the fusion add at 2 and 4 requests but interact at 16. If both go upstream, P needs a re-test on top of #69778 with the skinny table, at 16 requests and above. I have not run that combination.
- Greedy output with fusion: P identical to no-P.
