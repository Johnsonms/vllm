# B7 — can the shared expert get the SMs earlier? (B300, 2026-11-25)

Qwen3.6-35B-A3B-FP8, TP1, real weights. Both arms: e66bede56b + the `(10, 3)` table line (B3 `pr` arm). Edits were local only and are reverted (`~/vllm-kopt` clean at e66bede56b).

## Verdict
- **Priority (P) does reach the GPU through CUDA-graph replay.** No torch change is needed.
- **Order (O) is already in place.** The shared expert is enqueued before the router GEMM.
- **P is not a uniform win, so I don't recommend it as a blanket change.** In the paired A/B it helps at c1/c2/c8 and hurts at c4/c16, and the sign is the same in both GPU passes and in all 12 burst pairs per c:

| c | pass 1 (T on GPU 0) T / P ms, Δ | pass 2 (T on GPU 1) T / P ms, Δ | **mean Δ** |
|---|---|---|---|
| 1 | 2.931 / 2.876, −1.86% (6/6 pairs neg) | 2.914 / 2.903, −0.37% (6/6 neg) | **−1.1%** |
| 2 | 3.220 / 3.057, −5.07% (6/6 neg) | 3.198 / 3.073, −3.91% (6/6 neg) | **−4.5%** |
| 4 | 3.714 / 3.804, +2.42% (0/6 neg) | 3.713 / 3.807, +2.54% (0/6 neg) | **+2.5%** |
| 8 | 4.558 / 4.473, −1.86% (6/6 neg) | 4.550 / 4.470, −1.76% (6/6 neg) | **−1.8%** |
| 16 | 5.671 / 5.746, +1.31% (0/6 neg) | 5.673 / 5.721, +0.84% (0/6 neg) | **+1.1%** |

Values are TPOT p50, the mean of 6 bursts per arm. The per-pair ranges are in `paired_table.md`.

**Mechanism:** priority lets the shared expert take SMs as soon as they free up, but that slows the routed-MoE GEMMs it now competes with. It pays off where the shared expert was starving and is the longer branch, and costs where it already overlapped or where the routed MoE dominates.
- **c8 helps:** in T the shared expert still waits behind the MoE GEMMs, as in B6.
- **c4 hurts:** with the table on, the shared expert already overlaps at c4 (B6), so P only adds contention.
- **c16 hurts:** the routed MoE is the long branch there.

The profiled run predicted c4 +2.1% and c8 −1.9%; the paired A/B measured +2.5% and −1.8%.

**Correctness:** greedy tokens are identical on 4/4 prompts (64 tokens each) in both passes. GSM8K T 0.963 / 0.958 vs P 0.962 / 0.964. No tracebacks, no failed requests, AOT loads 0 in every server log.

## 1. Does stream priority survive CUDA-graph replay?
`prio_micro.py`: the main stream runs one SM-filling GEMM (8192³ bf16, ~630 µs). An aux stream waits on an event recorded before the GEMM, then runs 5 small kernels. Times are from torch-profiler kernel timestamps, median of 20.

| mode | aux work done after GEMM start |
|---|---|
| eager, aux priority 0 | 600 µs |
| graph, aux priority 0 | 589 µs |
| eager, aux priority −1 / −3 | 275 / 273 µs |
| **graph, aux priority −1 / −3 (torch `CUDAGraph.instantiate`)** | **135 / 145 µs** |
| graph, priority −3, re-instantiated by us with `cudaGraphInstantiateFlagUseNodePriority` | 144 µs |

- Captured kernel nodes carry the capturing stream's priority as their `cudaLaunchAttributePriority` (read back from the graph: 0 for the main-stream GEMM, −1 or −3 for the aux kernels).
- torch's own instantiate already gives the full effect: adding `UseNodePriority` changes nothing (144 vs 145 µs). On this stack (torch 2.14.0+cu132, driver 595.112.07, CUDA 13.2), no torch-side flag is needed.
- `torch.cuda.Stream.priority_range()` is (0, −3), so −1 is not the highest. It behaves the same as −3 here; any priority above the main stream's is enough. P used −1 as specified.

## 2. Order (variant O)
Qwen3.6 on CUDA runs through `vllm/models/qwen3_5/nvidia/model.py` (`Qwen3_5MoeForConditionalGeneration`), whose MoE layer is `Qwen3NextSparseMoeBlock`. That block calls the FusedMoE runner with `router_logits=hidden_states`, and the runner holds the gate. In `moe_runner.py` `_forward_impl`:
1. `self._shared_experts.maybe_forward_async(shared_experts_input)`: records an event on the main stream, then enqueues the whole shared expert on the aux stream;
2. then the router GEMM `self.gate(hidden_states)` on the main stream;
3. then routing and the routed-MoE kernels.

So the shared expert is **already** enqueued before the router; O would change nothing. The B6 timeline agrees: the shared expert's first kernel starts about 0.7 µs after the router, on the aux stream. Its delay comes later, from SM contention with the routed-MoE GEMMs. No O prototype was built.

## 3. Timeline with P (one profiled run per arm, B6 method, GPU 0)
| | busy union T → P (µs/step) | summed kernel T → P | overlap T → P |
|---|---|---|---|
| c4 | 3739 → 3818 (+79) | 5731 → 6027 | 1991 → 2209 |
| c8 | 4542 → 4455 (−87) | 6268 → 6806 | 1726 → 2351 |

- **c8** (`timelines/layer_{T,P}_c8.txt`):
  - In T, the shared expert's `act_and_mul` waits until +25 µs and its down-projection runs at +53 µs, after the routed finalize.
  - In P, `act_and_mul` runs at +15 µs and the down-projection at +21 µs, inside the routed down-GEMM window.
  - The routed gate-up GEMM gets slower (32 → 36 µs) and finalize goes 4.4 → 7.3 µs. Net: the layer ends 12 µs earlier in this sample; −87 µs/step overall.
- **c4** (`timelines/layer_{T,P}_c4.txt`):
  - In T the shared expert already finishes inside the routed-MoE window.
  - In P it runs concurrently with the routed down-GEMM, which slows 17.6 → 20.9 µs, and finalize goes 4.9 → 6.1 µs.
  - The layer ends ~0.6 µs later; +79 µs/step overall.

## Edit (described in words; nothing committed)
`aux_stream()` in `vllm/utils/torch_utils.py` created its stream with a priority read from an environment variable (unset = 0, the base behavior) and printed that priority once. Both arms loaded the same tree; T ran with the variable at 0 and P at −1. `code_checks.txt` has the per-server checks: loaded file paths, the `(10, 3)` table present, the edit present, and the priority actually used.

## Method (paired A/B, W12c design)
- Both servers up at once (B3 serve flags, `VLLM_DISABLE_COMPILE_CACHE=1`, launched from the tree).
- For each c in 1/2/4/8/16: 6 rounds, one burst per arm per round (5×c prompts, random 512/512, `--ignore-eos`, `--num-warmups c`), order flipped each round.
- Pass 1: T on GPU 0, P on GPU 1. Pass 2: swapped. Reported: mean of the per-arm burst means, and the mean Δ of the two passes.
- Each pass: greedy-token check (4 prompts, `temperature 0`, 64 tokens, token lists compared) and one GSM8K per arm.
- Other-user GPU check before every burst; it never fired.

## If the manager wants to pursue it
A batch-size gate (high priority only for decode batches ≤ 2 and around 8) would fit these numbers, but it is tuned to one model, one GPU and one table, so I'd treat it as overfitting unless W13 on B200 shows the same sign pattern. The B200 comparison (W13) decides whether there is a general rule.

## Files
`paired_table.md`, `paired_runs.json` (per-burst TPOT + greedy tokens), `prio_micro.txt`, `code_checks.txt`, `timelines/` (layer views and union/exposed for T vs P at c4/c8), `scripts/` (`prio_micro.py`, `b7_profile.sh`, `b7_paired.sh`, `analyze_paired.py`; the B6 analysis scripts are in `results/b300/B6/scripts/`).
