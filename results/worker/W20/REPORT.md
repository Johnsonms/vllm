# W20 — is the shared-expert gate on the critical path? (2026-11-28 07:08Z)

## Answer
**Yes. The gate runs on the side stream, but the side-stream branch is the one that reaches the join last, so its length sets when the layer can continue.** At 8 requests the shared-expert branch ends after the routed branch in 99.5 % of MoE layers on main (median 6.6 us later). The PR shortens the gate from 8.3 us of kernel time in two kernels to 2.4 us in one, the join moves 2.6 us earlier per layer, and the layer period drops by 2.7 us (2.3 %). That is the E2E gain.

The reviewer's premise holds only if the side branch has slack. Here it has none: the side stream is starved by the routed-expert GEMMs (the gate waits a median 15 us before it even starts at 8 requests), which makes the "overlapped" branch the longer one.

## Setup
- Qwen3.8-Flash-Next-FP8, real weights, TEP4 on 4 B200, custom all-reduce on, compile cache off (AOT count 0), no speculative decoding. Arms: main `361f362033` and PR `383e366bcb` (loaded paths and HEAD logged; gate plan empty on main, 11 sizes on the PR).
- torch profiler through `--profiler-config` (skip 24 engine steps, record 32), rank 0 trace, 512-token prompts, 8 and 4 concurrent requests. 1472 MoE layer executions per trace (32 steps x 46 layers).
- Script: `join.py` (kernel events only). Raw output: `join_output.txt`. One profile per arm and concurrency; the two arms are separate server starts.

## Code path (checked in the tree)
- With TP > 1 the model uses `Qwen4ExpSparseMoeBlock` (a `Qwen3NextSparseMoeBlock`), not the MegaMoE block (that one is TP1 only). It builds the routed experts through `FusedMoEFactory` and hands them the shared expert (`Qwen3NextMLP` with `expert_gate`).
- The MoE runner enqueues the shared expert on the aux stream (`SharedExperts.maybe_forward_async`, taken when the token count is under the stream threshold), runs router + routed experts on the main stream, then waits on the shared-output event and adds `shared_out`.
- Inside the shared expert the order is: gate/up GEMM, SiLU-and-mul, down GEMM, **then the gate `(1, 2560)` GEMM, sigmoid, multiply**. The gate is the third-last kernel of the side branch, so any wait or time in it delays the end of that branch directly.
- What the side branch overlaps on the main stream: router GEMM, FP8 quant, routing kernel, gate/up FP8 GEMM, activation, down FP8 GEMM, finalize.

## Per-layer numbers (us, medians; p25-p75 in brackets where useful)

| | main, 8 req | PR, 8 req | main, 4 req | PR, 4 req |
|---|---|---|---|---|
| shared branch ends last (share of layers) | **99.5 %** | 71.4 % | **66.4 %** | 24.9 % |
| shared end - routed end | +6.6 (+5.5...+7.4) | +3.9 (-0.2...+4.9) | +2.8 (-0.5...+4.3) | -10.9 (-11.4...-0.2) |
| gate kernels | nvjet splitK + splitKreduce | 1 skinny kernel | cuBLAS dot + reduce | 1 skinny kernel |
| gate kernel time | 8.3 | 2.4 | 4.2 | 2.2 |
| wait before the gate starts | 15.1 | 15.1 | 6.7 | 0.3 |
| down-proj end -> gate end | 30.6 | 17.4 | 13.1 | 2.4 |
| shared branch length | 48.2 | 43.9 | 36.6 | 24.1 |
| routed branch length | 48.0 | 48.1 | 39.8 | 39.3 |
| routed end -> join start | 6.9 | 4.3 | 3.2 | 0.4 |
| shared end -> join start | 0.3 | 0.5 | 0.5 | 11.2 |
| decoder layer period | 115.2 | 112.5 | 100.2 | 97.1 |
| TPOT of the unprofiled run on the same start (ms) | 6.100 | 5.985 | 5.295 | 5.256 |

How to read it:
- **The join follows the shared branch on main.** "shared end -> join start" is 0.3-0.5 us, while the routed branch has been finished for 6.9 us (8 requests) or 3.2 us (4 requests). The main stream is idle and waiting for the side stream.
- **The routed branch does not change** between the arms (48.0 vs 48.1, 39.8 vs 39.3). The whole difference is on the side branch.
- **8 requests:** the gate still waits 15 us to start on both arms (starved), but it then takes 2.4 us instead of 8.3, and one launch instead of two. The join moves from 6.9 to 4.3 us after the routed end (-2.6 us); the layer period moves by -2.7 us. 2.7 us x 46 layers = 124 us per step = 2.0 % of TPOT. W17 measured -1.67 % paired; this start pair shows -1.9 %.
- **4 requests:** on main the gate waits 6.7 us and the shared branch is last in two thirds of the layers. With the PR the gate starts at once, the shared branch ends about 11 us before the routed one in most layers, and the join now waits on the routed branch (0.4 us after it). The join moves by -2.8 us and the period by -3.1 us.
- At 4 requests the period change (3.1 us x 46 = 143 us = 2.7 % of TPOT) is larger than the paired TPOT change of W17 (-1.35 %). I do not have an explanation for that gap; the profiler is attached in one and not in the other, and the profiled window is 32 steps. The direction and the mechanism are the same.
- The remaining shared-last layers with the PR at 8 requests (71 %) are the 15 us start wait: the next limit there is the starvation of the side stream, not the gate kernel.

## Why 1 request shows nothing
Not profiled in W20. W14's 1-request trace of this model had the shared branch ending first, and W17 measured -0.2 % there, inside the spread. That fits: with one token the routed GEMMs are short and the side stream is not starved, so shortening the gate changes nothing at the join.

## Not checked
- Ranks 1-3 (only the rank-0 trace was analysed; all ranks run the same graph).
- Whether the same holds with speculative decoding or at TP8.
- One profile per arm and concurrency, no repeat.
