# STATUS — B300 node (role `b300`)

## B1 — ack 2026-11-24
- Read README, setup/B300_START.md, TASKS_B300.md. Starting B1.
- Node: 8x NVIDIA B300 SXM6 AC, driver 595.112.07, CUDA toolkit 13.2. All 8 GPUs idle (0 MiB, 0%) at start.
- Background host load ~10 (system daemons, ~9 of 256 cores); no user GPU jobs.
- Next: build env at e66bede56b, then sanity tests, coverage read, decode profile.

## B1 — progress 2026-11-24
- Env built at e66bede56b (precompiled wheel), FlashInfer cubins + jit-cache (incl. sm103a) downloaded.
- Versions: torch 2.14.0+cu132, FlashInfer 0.7.4, Triton 3.8.0, cuBLAS 13.4.0.1, NCCL 2.30.7, vLLM 0.34.1rc1.dev61+ge66bede56; capability (10, 3) confirmed.
- Model config/tokenizer for Qwen3.6-35B-A3B-FP8 downloaded by name (no token needed; dummy weights).
- Push access: OK.
- Running: `tests/kernels/test_bf16_skinny_gemm.py` (789 items, CuTe compile-bound, ~30 min). Decode profile follows after it, on an idle node.

## B1 — progress 2026-11-24 (2)
- Sanity: `pytest tests/kernels/test_bf16_skinny_gemm.py -q` → **351 passed, 438 skipped** (4m55s, GPU 0).
- Coverage (read only): Qwen4Exp SM103 table = 12 shapes / 34 entries, TP4 (6 shapes) + TP1 (6 shapes), no TP2/TP8 (SM100 has 24 shapes). Generic `SKINNY_GEMM_PLANS_BY_CAPABILITY` has only `(12, 1)`: no `(10, 3)`. `FI_ALLREDUCE_FUSION_MAX_SIZE_MB[103]` = {2: 64, 4: 64, 8: 4, 16: 64}.
- TP8 AR backend on this node: `Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl` (no trtllm fallback), Qwen3-235B-A22B dummy TP8, `allreduce_rms` fusion enabled.
  - Note: Qwen3.6-35B-A3B-FP8 cannot load at TP8 ("input_size_per_partition = 64 is not divisible by weight quantization block_k = 128"); expected for this checkpoint, so the check used the 235B config.
- Running: TP1 decode profile (c1, c8). Node rule from the owner: if anyone else uses the GPUs, this node stops its jobs and steps back.

## B1 — DONE 2026-11-24
Full report: `results/b300/B1/SUMMARY.md`.
- Env OK at e66bede56b, capability (10, 3). Skinny GEMM test: 351 passed / 438 skipped.
- SM103 coverage: Qwen4Exp SM103 = TP1 + TP4 only (12 shapes / 34 entries); generic skinny table has no (10, 3); AR fusion cap TP8 = 4 MB; TP8 FlashInfer AR resolves to **mnnvl**.
- Qwen3.6-35B-A3B-FP8 TP1 decode, median TPOT: **c1 3.00 ms, c8 4.92 ms** (2 runs each, within 0.1%).
- **cuBLAS BF16 GEMMs = 18.0% of kernel time at c1, 12.4% at c8** (B200: 15.9 / 13.3%). 221 launches/step: router, GDN `in_proj_ba`, shared-expert gate, LM head + 70-110 splitKreduce. Without the aux-stream shared-expert gate: 14.6% / 10.3%.
- Waiting for the manager's review before B2. Polling every ~10 min.

## B2 — ack 2026-11-24
- Read the B1 review. Starting B2: 14 W11 shapes, M ∈ {1, 2, 4, 8}, sweep then 4-seed × 3-round verify; also reporting whether each W11 SM100 config qualifies on SM103.
- Real-weight download of Qwen3.6-35B-A3B-FP8 started in the background (files by name).

## B2 — sweep done 2026-11-24
- Manager's `sweep_skinny_generic.py`, 8 processes (one per GPU: {small, LM-head} × M ∈ {1,2,4,8}), CUPTI, CUDA graph, cold L2. 56/56 cells, no errors. The weight download (host/network only) overlapped the sweep; the verify runs on a quiet node.
- **49 entries on 14 shapes pass ≥ 1.10× — exactly the same cells as W11 on SM100.** Dropped: every LM head at M=8, and (248320, 2048) / (124160, 2048) at M=4 (1.04× / 1.07×).
- Small shapes 1.56–3.28×; LM heads 1.13–1.44× (M=1–2), 1.13–1.23× at M=4 for ≤ 62080 rows.
- Only 8 of 49 winning configs equal the W11 SM100 config. The verify times cuBLAS vs SM103 proposal vs W11 SM100 config side by side (`verify_sm103.py`), 4 seeds × 3 rounds, GPUs 0-3.
- Raw: `results/b300/B2/sweep.jsonl`, `proposed.json`.

## B2 — DONE 2026-11-24
Full report: `results/b300/B2/SUMMARY.md` (+ `verified.json`, `verify_summary.json`, raw `verify.jsonl`).
- **All 49 SM103 entries verify** (4 seeds × 3 interleaved rounds, CUPTI, CUDA graph, cold L2, GPUs 0-3). Same 49 cells as W11.
  - Small projections 1.50–3.12×; LM heads 1.13–1.46×.
  - Error vs fp32 equals cuBLAS in all 196 records (max 3.8e-3).
- **Shared-table question:** all 49 W11 SM100 configs also pass ≥ 1.10× on SM103.
  - sm100_us / b300_us: median 1.000, range 0.973–1.049. The worst cells are LM heads at M=1 (4.9% and 3.4% slower).
  - Thinnest SM100 margin on SM103: (62080, 2048) M=4 at 1.11× (min 1.10).
  - Registering (10, 3) to the SM100 table is defensible; a separate table gains ≤ ~5% on a few LM-head cells.
- Real weights downloaded (35 GB). Ready for the E2E A/B when the manager's table is up. Polling every ~10 min.

## B3 — ack 2026-11-24
- Read the B2 review (shared table). Setting up arms: base = e66bede56b, pr = 81159c69d6 + local one-line (10, 3) registration (not committed anywhere). TP1 real weights, c = 1, 2, 4, 8, order base/pr/base/pr, GSM8K per arm.

## B3 — progress 2026-11-24
- TP1 set 1: a run-counter bug in my script (loop variable shared with the cleanup function) made round 4 (pr) overwrite round 2 (pr). Kept: base r1, base r3, pr r4. All three code checks are as expected, AOT = 0. GSM8K: 0.963 / 0.962 (base), 0.961 (pr).
- Fixed the script and rerunning a clean base/pr/base/pr set 2 now. Set 1 will be reported as supplementary.

## B3 — TP1 done 2026-11-24 (set 2, clean A/B/A/B)
- Median TPOT, mean of 2 runs per arm, base → pr: c1 3.000 → 2.928 (**−2.4%**), c2 3.339 → 3.231 (**−3.2%**), c4 4.182 → 3.703 (**−11.5%**), c8 4.762 → 4.584 (**−3.7%**).
- Per-run spread is ≤ 0.03 ms within each arm. Set 1 (3 surviving runs) agrees: −2.3 / −3.3 / −10.8 / −4.5%.
- GSM8K: base 0.964 / 0.958, pr 0.964 / 0.964. Code checks OK in every arm; AOT = 0.
- c4 stands out: base c2→c4 adds 0.84 ms vs 0.58 ms c4→c8. That suggests a slow cuBLAS choice at M=4 on SM103; not profiled yet.
- Running the optional TP2 A/B (c1, c8) now.

## B3 — DONE 2026-11-24
Full report: `results/b300/B3/SUMMARY.md` (+ `runs.json`).
- TP1 (clean A/B/A/B): TPOT **−2.4% c1, −3.2% c2, −11.5% c4, −3.7% c8**.
- TP2 (optional): **−3.8% c1, −5.1% c8**. The two c1 base runs differ by 3%.
- GSM8K unchanged in every arm (0.958–0.964, 0 invalid). Code-check lines and AOT = 0 confirmed in all 12 runs.
- The c4 gain at TP1 is unusually large (base is slow at batch 4 on SM103). Not profiled; I propose a c4 profile of both arms if the manager wants the cause.
- Idle; polling every ~10 min.

## Manager update (11-25) — ack 2026-11-25
- Read the 11-25 update. Node idle (8 GPUs, 0 MiB). Polling resumed (~10 min).
- Queue: **B4** (SM103 verify of dense-Qwen3 91 entries + Qwen3-MoE 49 entries, cuBLAS vs given config, 4 seeds × 3 rounds, one seed per GPU) → B5 (Qwen3-32B TP8 AR-fusion cap 4/8/16/32 MB) → B6 (c2/c4/c8 decode profile, base vs (10, 3) table).
- Starting B4 now.

## B4 — DONE 2026-11-25
Full report: `results/b300/B4/SUMMARY.md`.
- 140 cells (dense Qwen3 41 shapes / 91 entries + Qwen3-MoE 15 / 49), cuBLAS vs the given B200 config, 4 seeds × 3 rounds, CUPTI + CUDA graph + cold L2, one seed per GPU on an idle node.
- **0 cells below 1.00×.** 134 ≥ 1.10×; 6 dense cells in 1.05–1.10× (1280,5120 M8; 5120,1024 M2; 5120,1280 M2; 5120,2048 M2; 151936,5120 M1/M2), each with seed spread ≤ 0.02. Qwen3-MoE: all 49 ≥ 1.13× (up to 2.49×). Error = cuBLAS everywhere.
- Per the decision rule: no drop, no SM103 split needed.
- Starting B5 (Qwen3-32B weights already downloaded by name).

## B5 — progress 2026-11-25
- Read the B4 review. B5 running: Qwen3-32B real weights, TP8, idle node.
- DEBUG probes (one launch per arm, untimed): cap 4 / 8 / 16 / 32 MB → `Maximal number of tokens used by Flashinfer Allreduce Fusion` = **409 / 819 / 1638 / 3276**; backend **mnnvl** in every arm; `allreduce_rms` fusion enabled; AOT loads 0.
- Note: at 409 tokens the default cap already covers every pure-decode batch up to c384, so a larger cap can only change steps that carry a prefill chunk. Expect a smaller effect than B200's 1 MB → 16 MB.
- Timed round 1 done for default, 8, 16; round 2 follows. Then B6.

## B5 — DONE 2026-11-25
Full report: `results/b300/B5/SUMMARY.md`.
- **Pick: keep 4 MB.** At caps of 8, 16 and 32 MB, TPOT at c32–c256 is within ±2% of default, with mixed signs; GSM8K 0.941–0.945 in all arms.
- Reason: on SM103 the default 4 MB already covers 409 tokens, which includes every decode batch up to c384. B200's #69543 gain came from moving c128+ out of an unfused 1 MB (102-token) cap.
- c384 is noisy (runs of about 16 s; up to 3 ms between rounds of the same arm). The −4.9% at 32 MB rests on one outlier run, so I don't count it.
- One consistent effect: 1k-prefill TTFT at c4 is −4.7% with 32 MB in both rounds. A prefill-heavy A/B of 4 vs 32 MB is a possible follow-up; it is not part of this pick.
- Probes: 409 / 819 / 1638 / 3276 tokens; backend mnnvl; AOT 0 everywhere.
- B6 started (base then pr, c2/c4/c8 profiles, GPU 0).

## B6 — DONE 2026-11-25
Full report: `results/b300/B6/SUMMARY.md`.
- **The c4 step is not a cuBLAS kernel.** cuBLAS BF16 time per step is flat from c2 to c4 (789 → 780 µs). In both arms the largest increase is the routed-MoE FP8 `bmm_*` GEMMs: +530 µs/step, with a tile switch from t128x16 s6 to t128x8 s8, as the number of distinct experts grows from 15.5 to 30.1.
- **What is base-only at c4 is lost overlap.** The shared expert on the aux stream gets serialized behind the routed-MoE GEMMs. Exposed time at c4: MoE gate-up 370 vs 17 µs/step; cuBLAS shared-gate GEMV 180 vs 0. The GPU busy union matches TPOT within 0.1 ms; base → pr at c4 is −454 µs, of which +202 µs is recovered overlap. pr's single-kernel router and gate let the shared expert start early enough to run alongside the MoE.
- At c8 the shared expert waits behind the MoE in **both** arms, so pr gains only kernel time there. Possible follow-up (not done): launch the shared expert earlier, or give the aux stream priority.
- cuBLAS names on SM103 at M = 2/4/8 are listed in the report. The router switches cutlass_80 → `nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN` exactly at M=4; the shared gate switches GEMV → nvjet split-K at M=8.
- B4, B5 and B6 are done. Tree restored to base; GPUs free. Polling every ~10 min.

## B7 — ack 2026-11-25
- Read the B6 review. Starting B7: (1) does aux-stream priority survive CUDA-graph capture/replay (variant P); (2) where the router runs relative to `maybe_forward_async` and whether the shared expert can be enqueued earlier (variant O); (3) paired A/B (two servers on two GPUs, alternating bursts, GPU swap), c = 1/2/4/8/16, GSM8K × 2, greedy-token identity check.
- Both arms = e66bede56b + `(10, 3)` table line; each arm runs from its own copy of the tree with PYTHONPATH set, and logs the loaded file paths and edit markers. Local edits only.

## B7 — interim 2026-11-25
- **Priority survives CUDA-graph replay here.** Micro test: one SM-filling GEMM on the main stream (8192³ bf16, ~630 µs) plus 5 small kernels on an aux stream that waits on an event recorded before the GEMM.
  - With a default-priority aux stream, the aux work finishes 590–600 µs after the GEMM starts, in eager and graph mode alike.
  - With priority -1 or -3: 248–275 µs eager, **135–145 µs under graph replay**.
  - Captured kernel nodes carry the stream's priority as their launch attribute (-1 / -3). torch's own `CUDAGraph.instantiate()` already honors it; instantiating the same graph with `cudaGraphInstantiateFlagUseNodePriority` gives the same result (139 vs 143 µs). No torch change needed. torch reports the priority range as (0, -3); -1 and -3 behave the same.
- **Variant O is already how the code works.** Qwen3.6 on CUDA goes `vllm/models/qwen3_5/nvidia/model.py` → `Qwen3NextSparseMoeBlock` → FusedMoE runner `_forward_impl`, which calls `maybe_forward_async` (shared expert onto the aux stream) *before* the router GEMM (`self.gate(...)`, the runner holds the gate). The B6 timelines agree: the shared expert's first kernel starts 0.7 µs after the router, on the aux stream. The problem is SM contention later, not launch order. No O prototype.
- **P in the server (one profiled run per arm, table on in both, GPU 0):**
  - Overlap rises: c4 1991 → 2209, c8 1726 → 2351 µs/step.
  - Contention slows the routed-MoE kernels too: summed kernel time +296 / +538 µs.
  - Net busy union: c4 +79 µs (+2.1%), c8 −87 µs (−1.9%).
  - In a c8 layer, the shared expert now finishes inside the MoE down-GEMM window (+37 vs +66 µs) and the layer ends 12 µs earlier.
- Edit (local only, described in words): `aux_stream()` creates its stream with a priority read from an env var (unset = 0); the `(10, 3)` table line is in both arms. Both servers load the same tree and log the priority they used.
- Running: paired A/B T vs P (GPUs 0+1, then swapped), c = 1/2/4/8/16, GSM8K × 2 per arm, greedy-token check.

## B7 — DONE 2026-11-25
Full report: `results/b300/B7/SUMMARY.md`.
- **Priority survives CUDA-graph replay** on torch 2.14 / CUDA 13.2. Captured nodes carry the aux priority; torch's instantiate honors it without `UseNodePriority`.
- **Order (O) is already in place.** `maybe_forward_async` runs before the router GEMM in `moe_runner._forward_impl`, so no prototype was needed.
- **P (aux priority −1), paired A/B with the GPUs swapped, mean of 2 passes:** c1 −1.1%, c2 −4.5%, **c4 +2.5%**, c8 −1.8%, **c16 +1.1%**. The sign is the same in both passes and in all 12 pairs per c.
- Greedy tokens identical (4/4 per pass). GSM8K T 0.963/0.958 vs P 0.962/0.964.
- Mechanism: priority helps where the shared expert was starved (c8, as in B6). It hurts where the shared expert already overlapped (c4 with the table on) or where the routed MoE is the long branch (c16), because the routed-MoE GEMMs slow under contention.
- **Not recommended as a blanket change.** A batch-size gate would fit these numbers but looks like overfitting unless W13 on B200 shows the same pattern.
- Local edits reverted; tree clean at e66bede56b; GPUs free. Polling every ~10 min.

## B8 — ack + step 1 2026-11-25
- Read the B7 review and B8. Trees: git worktrees at `fb159244e6` (base) and `c5ef00f633` (#69778), with the prebuilt ops copied in (no csrc change between e66bede56b and fb159244e6). Each server runs from its own tree with `PYTHONPATH` set.
- **Step 1 — the fusion fires on SM103.** In-process load of Qwen3.6-35B-A3B-FP8, `layers.0.mlp.shared_expert.down_proj`:
  - base: `Fp8LinearMethod` / `CutlassFp8BlockScaledMMKernel`, `_input_quant_key` = None.
  - PR: same classes, `_input_quant_key` = `QuantKey(float8_e4m3fn, scale fp32 dynamic, group (1, 128), symmetric)`, matching the kernel's `input_quant_key()`.
  - Profiled decode of 4 prompts × 32 tokens: base runs `act_and_mul_kernel` ×1280 plus `per_token_group_quant…<true,false,128>` ×5120. PR runs `silu_and_mul_per_block_quant_kernel` ×1280 instead, and that quant kernel drops to ×3840, so one shared-expert `act_and_mul` + quant pair per layer per step is replaced.
- Running: paired A/B (c = 1/2/4/8, GPUs 0+1 then swapped) and GSM8K 8 runs per arm, interleaved, with per-question dumps.
- Note: the owner has also given this node a local kernel-optimization task for the next ~12 h (local commits only, nothing pushed). It uses one spare GPU for unit tests and microbenchmarks, and never the GPUs a B-task is timing on.

## B8 — DONE 2026-11-25
Full report: `results/b300/B8/SUMMARY.md`.
- The fusion fires on SM103 (`_input_quant_key` = group-128 FP8 on the shared-expert `down_proj`; `act_and_mul` + one quant per MoE layer → `silu_and_mul_per_block_quant_kernel`).
- Paired TPOT (2 GPU-swapped passes): c1 **−2.25%**, c2 −0.13% (passes disagree), c4 **−1.44%**, c8 **−2.24%**.
- GSM8K, 8 runs per arm: base **0.9636** (sd 0.0026) vs PR **0.9612** (sd 0.0020), −0.0024 (≈3/1319, p ≈ 0.06; same sign as B200's −0.0041).
  - Flips base→PR 20.6 per run pair vs A/A floor 16.5: about 4 extra right→wrong, all among 54 unstable questions.
  - 0 questions right in every base run and wrong in every PR run.
- **Cause of any difference: the fused kernel is not bit-identical to the unfused chain** (greedy differs on 3/4 prompts). It quantizes the fp32 `silu(gate)·up` without the bf16 rounding the unfused chain does, floors the scale with `min_scaling_factor` instead of eps on absmax, and divides instead of multiplying by 1/scale.
  - Suggestion (needs a csrc build, untested): round to the input dtype before absmax/quant and match the other two details. That makes it bitwise the unfused chain and closes the accuracy question by construction.
- Node: the owner's local kernel work continues on this node (local only). Polling every ~10 min.

## Note on the bitwise fused producer — 2026-11-25
- Read the B8 review. To avoid duplicate work: a bitwise variant of `silu_and_mul_per_block_quant` already exists on this node as a **local commit on the owner's branch** (not pushed; the owner decides how to share it).
- What it does, in words. Both kernels (one block per group, and vectorized) use the unfused arithmetic:
  - SiLU as `gate / (1 + expf(-gate))`, rounded to the input dtype; the product with `up` rounded to the input dtype;
  - group absmax floored at 1e-10, `scale = absmax / 448`, the optional `scale_ub` clamp kept;
  - quantize as `x * (1 / scale)`, with one reciprocal per group.
- Verified on SM103 with the file JIT-built against the tree's headers (sm_103a). On random inputs (M up to 4000; both kernel paths; all-zero and outlier groups) FP8 codes and scales are **identical** to `silu_and_mul` + `per_token_group_quant_fp8` (column-major, no UE8M0); the shipped kernel differs in ~3% of codes and almost all scales.
  - A new bitwise test fails 16/16 on the shipped kernel and passes 16/16 on the fix; the rest of the file's tests pass.
  - Speed unchanged (M=1 H=512: 1.92 → 1.89 µs; M=4096 H=512: 7.39 → 7.04 µs).
  - INT8 keeps rounding to nearest; the standalone INT8 group quant truncates.
- B300 continues on the owner's local kernel work.

## 11-25 ack: PR review request + PR_RULES.md
Read both. Nothing will be opened. Plan: (1) diff my local bitwise producer against `152bca1e65` and run my bitwise test on your kernel; (2) duplicate search for each local candidate; (3) post `results/b300/PRREVIEW/SUMMARY.md` once the final paired E2E (E8 TP2, E9 TP1, E10 full stack; pass 2 rerun after a client hang) finishes, about 1 h.

## 11-25 evening: PR review request answered
`results/b300/PRREVIEW/SUMMARY.md`. One candidate: skinny block-FP8 GEMM + TRT-LLM routing on SM103 (E2E TP1 −18.0/−16.4/−13.5/−7.0%, TP2 −17.2/−15.5/−11.6/−6.5% at c1/2/4/8; GSM8K 8×8 Δ +0.0013 at the flip floor). My bitwise SiLU producer will not be opened: `152bca1e65` is the same FP8 arithmetic except −0 inputs, where yours is the correct one (your test fails 12/443 on mine, passes 443/443 on yours; my test passes 363/363 on yours). Waiting for your review.

## 11-25 reply: overlap question + review ack
- **Overlap (one line):** yes. The owner-local `kopt/norm-group-quant-fusion` (1 commit on #69778 `c5ef00f633`) already adds group-128 (`kFp8Dynamic128Sym`) to the input Gemma add-RMSNorm (column-major scales), the GDN gated RMSNorm → `out_proj` (head_v_dim 128 = one group), and the TP>1 all-reduce + norm path. It does **not** cover the attention sigmoid gate → `o_proj`. Paired E2E TP1 vs #69778: −3.8 / −4.3 / −2.4 / −3.3% TPOT at c1/2/4/8, bit-identical greedy and tests. Your call whether to stop the B200 prototype; the owner decides whether this becomes a PR.
- **Review:** accepted, all three blockers. Starting (a) now: #69949 `301212073e` on SM103 (test with (10, 3), rule microbench vs my plan table on the Qwen3.6 shapes, paired A/B c1–c64 vs main). Then (b)'s evidence, measured against #69949.

## 11-25 18:55 ack: norm + quant findings
Read both notes. Honest answer to "how was bit-identical greedy measured" on `kopt/norm-group-quant-fusion`:
- Only the 4-prompt × 64-token check per pass, with short prompts (M ≤ ~12 at prefill), plus the unit tests (`torch.equal` at decode-sized M).
- So it does **not** cover your finding 1 (Triton vs FlashInfer Gemma norm at prefill M) or finding 2 (GDN at 4 rows per block). My "bit-identical" claim should be read as "at decode sizes" until measured.
- My reference test already passes `use_ue8m0=False` (finding 3).

**Queued** once the GPUs free up (~19:30; (a)/(b) E2Es are running now):
1. The 32 single + 256 batched × 256-token greedy counts vs #69778, including a base-vs-base floor.
2. An elementwise check of both sites at M = 300 / 2000 / 8192.
3. A paired E2E c1–c32 with your method.

## 11-25 21:20: (a)/(b) E2E done, norm+quant answers
- (a) #69949 on SM103 vs main, TP1 c1–c64: −10.0 / −8.7 / −5.4 / −4.4 / −4.0 / −3.3 / −2.1%, 84/84 pairs faster, AOT 0. (Kernel results already in `A_PR69949_SM103.md`.)
- (b) skinny on top of #69949: TP1 −10.1 / −8.7 / −8.5 / −2.9% at c1–c8, flat c16–c64; TP2 −6.4 / −8.4 / −4.4 / −1.5 / −4.1% at c1–c16, flat c32/c64. Greedy + GSM8K in `results/b300/PRREVIEW/B_skinny_on_69949.md`.
- Norm + quant: **correction** — not bit-identical at prefill sizes (your findings 1 and 2 reproduce on SM103). Greedy single 3/32 identical. Paired E2E vs #69778 c1–c32: −4.2 / −4.2 / −3.0 / −2.9 / −2.4 / −2.5%, 72/72. GSM8K 8×8 running. Details in `results/b300/PRREVIEW/NORMQUANT_B300.md`.
- Also on B300, owner-requested side work: MSA (MiniMax-M3) sparse prefill on SM103 (hardware ex2 + tcgen05.ld.red), local only.

## 11-25 22:05: reply to the manager review (3ced14b)
1. **#69949+SM103 single-prompt greedy vs main (SM103), 32 × 256 tokens: 32/32 identical**; base-vs-base floor 32/32. (Batched with the old client: 78/256 vs a 52/256 floor; see 2.) AOT 0 in both logs.
2. **Batched greedy floor.** Cause found: my old client sent the 256 prompts as 256 separate chat requests from a 256-thread pool, so arrival was async and batch composition differed per run. New client (`greedy_check2.py`): the 256 prompts (chat template applied, thinking off) go as **one** `/v1/completions` request with a prompt list, `max_num_seqs 256`. Rerun of #69949+SM103 vs main:
   - B1 vs A2: **256/256 identical**; single 32/32.
   - A1 (the first batched request a fresh server sees, run concurrently with B1 on the other GPU) differs from both: 109/256. So the batched set is deterministic except, apparently, on a server's first large request. I will use "2 batched runs per arm, compare the second" from now on unless you prefer otherwise.
3. (b) held as asked.
4. Norm + quant **G4 GSM8K 8×8 vs #69778: 0.9594 → 0.9646 (+0.0051, Welch t +6.1)**; flips A→B 21.2 per pair vs floors A/A 20.4, B/B 15.4; right-in-all-A & wrong-in-all-B: 0 (reverse: 1). Yes please, describe the attention sigmoid gate → `o_proj` kernel in words; I will add it after the owner's current B300 task.

## 11-25 22:15: G4 confounding check (b3b8ebf)
Reported as "no drop" for now. Answers:
- **GPUs:** A (base) on GPU 5 and B on GPU 6 for all 8 runs; **no swap**. That is a confound.
- **Timing:** both servers up at once; each of the 8 rounds runs both arms, order alternating (A B, B A, ...).
- **Base:** **not** `152bca1e65`. Base was `c5ef00f633` (#69778 first commit), and the norm+quant branch `ce0594a919` sits on that same commit, so both arms carry the same pre-fix SiLU kernel and the A/B isolates the norm change only. E14 (E2E) used the same pair.
- Plan: rerun GSM8K 2 × 4 runs with the GPUs swapped between halves, on the same pair. I also cherry-picked the branch cleanly onto `152bca1e65` (`kopt/norm-group-quant-fusion-on152`), but running that needs a `_C` rebuild for the `.cu` change in `152bca1e65`; I will do that only if you want the base to be exactly #69778's head.

## 11-25 22:45: norm + quant redo on `152bca1e65` started
- `_C_stable_libtorch` rebuilt from source on `152bca1e65` (note: with CUDA 13, `TORCH_CUDA_ARCH_LIST="10.0a 10.3a"` is filtered to 10.0a and fails on SM103 with "no kernel image"; `"10.0"` works). #69778's own test: 443 passed on SM103.
- Branch `kopt/norm-group-quant-fusion-on152` (owner-local) now has two commits on `152bca1e65`: the norm sites (cherry-pick, clean) and **the attention sigmoid gate → `o_proj` site**, implemented from your description (Triton `fused_sigmoid_gate_mul_group_quant`, column-major scales, wired in `Qwen3NextAttention` on `kFp8Dynamic128Sym`). New test: `torch.equal` on q and scale vs `fused_sigmoid_gate_mul` → `per_token_group_quant_fp8(column_major_scales=True, use_ue8m0=False)`, M ∈ {1, 7, 64, 300, 2048}, hidden 128/4096, bf16/fp16, contiguous and row-strided gate: 40/40. Related tests: 665 passed.
- Running now, base `152bca1e65` vs the branch: greedy (new client, one throwaway batched request after start, single 32 + batched 256), GSM8K 2 × 4 runs with GPUs swapped between halves, paired E2E c1–c32. ETA ~1.5 h.
- Side result (owner task): `tcgen05.ld.red` in FA4's MLA kernels gives no measurable gain on B300 (decode/prefill/sparse/SplitKV within ±1%); not pursued. MSA sparse prefill with hardware exp2 + ld.red on SM103: −8.6…−14.2% at kernel level, owner-local.

## Normquant redo — DONE 2026-11-25
Details: `results/b300/PRREVIEW/NORMQUANT_B300.md` (top section; old run kept below).
- A = `152bca1e65`, B = A + norm sites + attention gate → `o_proj` site (owner-local, 2 commits).
- Paired E2E (GPU-swapped, AOT 0): **c1 −4.95%, c2 −4.48%, c4 −3.14%, c8 −3.67%, c16 −3.33%, c32 −2.92%**; 72/72 pairs faster.
- GSM8K 2 × 4 runs, halves GPU-swapped: A 0.9621 (sd 0.0021), B 0.9615 (sd 0.0007), Δ −0.06 pp (t −0.50). Flips A↔B 20.0 vs floors 15.3 / 17.5; 1 prompt right-in-all-A / wrong-in-all-B, 0 reverse.
- Greedy: floor 32/32 single, 256/256 batched; A vs B 3/32 and 23/256 (expected: input-norm sites not bit-identical at prefill M).
- No vLLM code pushed; waiting for the manager's review. GPUs 1–6 free again.

## Normquant review (caddf9c) — ack + two early findings 2026-11-26
- **GSM8K is already 8 per arm.** Each half of G5 ran 4 per arm; my earlier post analyzed only h1. Over both GPU-swapped halves:
  - A 0.9612 (sd 0.0021), B 0.9606 (sd 0.0026), Δ −0.07 pp (t −0.56).
  - Flips A↔B 20.4 per pair vs floors A↔A 16.2 and B↔B 17.2.
  - Right in all 8 A and wrong in all 8 B: 0; reverse: 0.
  - Correcting the h1-only numbers in my last post.
- **Duplicates.** #65497, #36016 and #45770 change only the torch.compile `rms_quant_fusion` pass and the C++ norm-quant kernels it calls. None of them touches `vllm/models/qwen3_5/`.
  - Qwen3.5/3.6 runs with compile mode NONE: breakable cudagraph is auto-enabled for its architectures, and #51012 deprecated torch.compile for Qwen3.5. The server log shows `CompilationMode.NONE`.
  - So none of them fires on Qwen3.6 with this branch reverted, and #45770 changes nothing for Qwen3.5 on SM100. #65497's own body says "Qwen3.5 on NVIDIA is not affected: since #51012".
- **New overlap: #69812** (opened 2026-11-25 by lnkerr, open), "[Perf][Qwen3.5] Fuse dynamic per-token FP8 input quant into GDN gated RMSNorm and attention sigmoid gate".
  - It is the same two producers and the same files as ours: `fused_sigmoid_gate_mul.py` +91, `qwen3_next.py`, `fused_gdn_gated_norm.py`, `nvidia/model.py`, `test_gdn_output_projection.py`.
  - It covers `kFp8DynamicTokenSym` only, so it does not remove our group-128 launches on block-FP8 Qwen3.6. It will textually conflict with our GDN and gate commits, though.
  - Options: rebase on it and add the `kFp8Dynamic128Sym` epilogue to its kernels, or keep ours and note it. Your call / the owner's.
- Next: unit tests on base and branch (running), kernel microbench for the three sites (§4.3 — missing so far), then the PRREVIEW.

## B1 (input-norm-only PR) — ack + started 2026-11-26
- Ack the owner's split. GDN and gate are parked; there is no comment on #69812.
- **B1 branch** (owner-local): one commit on `152bca1e65`, 4 files, +134/−12.
  - Contents: `fused_norm_quant.py` (group-128 epilogue), `model.py` (input-norm key only, no GDN), the AR path, and the test.
  - Change vs the cherry-picked commit: the AR path now passes `use_ue8m0=False` explicitly. Before, it used the default, which follows `is_deep_gemm_e8m0_used()`. The CUTLASS linear's own `QuantFP8` sets False, so with DeepGEMM e8m0 on, the AR path would have produced different scales.
- **Unit tests, base `152bca1e65` vs B1** (same 4 files: `test_gemma_add_rms_norm_quant`, `test_fused_quant_layernorm`, `test_fused_sigmoid_gate_mul`, `test_gdn_output_projection`):
  - base 819 passed; B1 855 passed, which is the +36 new group-128 tests.
  - The new tests run against base code: 36 failed.
  - TP2 `test_fused_allreduce_norm_quant`: 1 passed on both. Pre-commit is clean.
- **Microbench, input-norm site.** Qwen3.6 H=2048; the FlashInfer gemma add-RMSNorm + `per_token_group_quant_fp8` vs fused; CUDA graphs, L2 flushed, 7 interleaved rounds × 3 GPUs.
  - 1.83–2.55× at M 1–128; 1.50× at 512; 1.34× at 2048; 1.25× at 8192. No shape is slower.
- **Running B1 vs `152bca1e65`:** paired E2E c1–c32 (GPUs 5/6), GSM8K 8+8 GPU-swapped (1/2), greedy single + batched (3/4). ETA ~1.5 h.
- **Side finding on the parked gate kernel.** My first `fused_sigmoid_gate_mul_group_quant` was 0.52× (slower) at M ≥ 2048: it launched one 1-warp program per group. Fixed locally, with one row per program above 64 tokens: now 1.83–2.41× at all M. Not part of B1.
  - The parked branch also needs a mypy fix in `fused_gdn_gated_norm.py` (`group_scale` Optional narrowing). It is noted for when #69812 lands.

## Ack 2d5cc9bc01 (owner delegates opening) — 2026-11-26
- Read and understood: B300 posts `PRREVIEW/<name>/` (`SUMMARY.md`, `title.txt`, `body.md`, `diffstat.txt`, branch + head SHA).
  - The manager posts `APPROVED TO OPEN: <name> @ <sha>`; only then do I open, exactly that SHA and body, from the owner's fork.
  - Before pushing I check author/committer identity and that there is no Co-authored-by.
  - No labels and no @-mentions.
  - Replies, comments (incl. #69812), force-pushes and closing stay owner-only; I draft those to the manager.
- Order: B1 → MSA → skinny; GDN + gate parked. The #69949 SM103 item is removed from my list.

## B1 evidence — done (B1 = `e77ca1439c` vs `152bca1e65`)
- **GSM8K.** The first 8+8 (GPU-swapped halves) was low: A 0.9628, B 0.9610, Δ −0.18 pp, t −3.2. Before reading anything into it, I ran a second 8+8 on fresh servers:
  - second set: A 0.9608, B 0.9615, Δ +0.08 pp (t +0.70);
  - **all 16+16: A 0.9618 (sd 0.0021), B 0.9612 (sd 0.0016), Δ −0.05 pp (t −0.81)**;
  - flips A↔B 23.4 per pair vs floors 17.1 (A↔A) and 16.2 (B↔B);
  - right in all 16 A and wrong in all 16 B: 0; reverse: 1.
  - The first set's gap is the same size as the base-vs-base shift between the two sets (A 0.9628 vs 0.9608). I read it as run-to-run noise, not a B1 effect, so I did not stop. Tell me if you want more.
- **Greedy** (warm-up client): base-vs-base 32/32 single and only 176/256 batched this time; the warm-up does not always make batched deterministic. A vs B: 3/32 and 19/256, the same as the three-site branch.
- **Paired E2E** (TP1 Qwen3.6-35B-A3B-FP8, 2 GPU-swapped passes × 6 rounds, AOT 0 in all 4 logs):

  | c1 | c2 | c4 | c8 | c16 | c32 |
  |---|---|---|---|---|---|
  | −2.56% | −1.92% | −0.70% | −1.70% | −0.78% | −1.31% |

  - 72/72 pairs are faster; the two passes agree within 0.2 pp.
  - The ablation answer: B1 alone gives about half of the three-site branch (−4.95 / −4.48 / −3.14 / −3.67 / −3.33 / −2.92%). The rest waits for #69812 plus our group-128 follow-up.
- Next: the B1 PRREVIEW package (`PRREVIEW/B1_input_norm_group128/`), then the MSA dup check.

## B1 PRREVIEW revised per c759645 — 2026-11-26
- `body.md` changes:
  - the greedy wording is now "(batched greedy was not reproducible run to run in this setup)";
  - Related PRs: added the #61073 / #63202 bullet. I checked both: neither touches `vllm/models/qwen3_5/` or the Gemma norm.
- Head unchanged: `e77ca1439c`. Footer: waiting for the owner's decision; I will not open before it.
- Noted: the rebase after #69778 merges needs a force-push, so I will ask then.
- Also posted: `PRREVIEW/MSA_sm103_prefill/`. Before opening it needs (1) OK to create the owner's MSA fork and (2) the same footer decision.
- Starting skinny: rebasing `kopt/skinny-on-69949` onto #69949 head `d6bccd2252`, then GSM8K 8+8 vs #69949 and batched greedy.

## Opened per 24b9aa2 / fb0b388 — 2026-11-26
- Ack owner decisions: footer kept (also for skinny later); MSA fork approved; order B1/MSA → skinny → W15 → GDN+gate.
- **B1_input_norm_group128 → vllm-project/vllm#70028**, draft, head `e77ca1439c`, from `Johnsonms:qwen35-input-norm-group128-quant`. Body = 3e907ed body + footer; live body verified.
- **MSA_sm103_prefill → vllm-project/MSA#53**, non-draft, base `dev` (487773e), head `311fa9d`, from `Johnsonms:sm103-prefill-softmax` (fork created per owner OK). Body = b1761d6 body + AI-assistance section + footer; live body verified.
- Identity checked before each push: author and committer = owner, Signed-off-by, no Co-authored-by. No labels, no mentions, no comments.
- Details: `PRREVIEW/<name>/opened.md` and `body.final.md`.
- B1 → ready only after #69778 merges; the rebase needs a force-push → will ask through you then.
- Skinny: GSM8K 8+8, batched greedy and paired E2E vs #69949 head `d6bccd2252` are running (all 8 GPUs). Unit tests: base 1 failed / 368 passed, skinny 1 failed / 396 passed. The 1 failure on both sides is the same pre-existing SM103 cuBLAS `test_w8a8_block_fp8_torch_scaled_mm_matmul` (CUBLAS_STATUS_NOT_SUPPORTED). PRREVIEW follows when the runs finish.

## Skinny evidence vs #69949 head `d6bccd2252` — **regression at c8, investigating** (2026-11-26)
Skinny = `0ed623973b` (cherry-pick of `6a11c99a04`, unchanged plans). Qwen3.6-35B-A3B-FP8 TP1, paired E2E (2 GPU-swapped passes × 6 rounds, AOT 0 in all 4 logs):

| c | pass 1 | pass 2 | mean | pairs faster |
|---|---|---|---|---|
| 1 | −9.68% | −9.57% | **−9.62%** | 12/12 |
| 2 | −8.49% | −8.47% | **−8.48%** | 12/12 |
| 4 | −3.11% | −3.24% | **−3.18%** | 12/12 |
| 8 | +2.75% | +2.71% | **+2.73%** | **0/12** |
| 16 | −0.00% | −0.16% | −0.08% | 7/12 |
| 32 | −0.28% | +0.13% | −0.07% | 8/12 |

- On the old #69949 base (`301212073e` + SM103 enable), E12 had c8 −2.85% (12/12 faster). So on the new head, skinny at M≈8 now loses, and c4 shrank from −8.45% to −3.18%.
- Suspect: the M=8 plan entries ((1024,2048) and (512,2048)) and possibly the M=4 entries, tuned against the old TRT-LLM path. Next: microbench the plan entries at M ∈ {3..8} against `d6bccd2252`'s dispatch, then drop or retune the losing entries and re-run E2E. **The PRREVIEW is held until this is resolved.**
- GSM8K 8+8: A 0.9629 (sd 0.0023), B 0.9613 (sd 0.0011), Δ −0.15 pp, Welch t −1.69. Flips A/B 16.9 vs floors 16.4 (A/A) and 19.3 (B/B); 0 systematic either way. Within noise.
- Greedy: single 3/32 identical (floor 32/32; the accumulation order differs); batched 222/256 (= floor 222/256).
- Unit tests: base 1 failed / 368 passed; skinny 1 failed / 396 passed. The same pre-existing SM103 cuBLAS failure on both sides.

### Skinny c8 investigation — step 1 (kernel level)
- Re-ran the rule-vs-plan microbench on the current trees (CUPTI, CUDA graph, cold L2, 4 seeds × 3 rounds). Every plan entry still beats #69949's TRT-LLM path: 1.19–2.31× at M 1/2/4, 1.38× (1024,2048) and 1.70× (512,2048) at M 8. TRT-LLM and skinny µs both match the 11-25 run within ±1%. FlashInfer is unchanged (0.7.4), and both trees use the same `_C`.
- So the GEMMs are not the cause. The base arm got 5.7–6.7% faster at c4/c8 from the newer main merged into `d6bccd2252` (E12 base 3.963/4.589 ms → E18 3.699/4.326). The skinny arm did not get that gain (3.618/4.449 → 3.584/4.445).
- Next: profile one decode step at batch 4/8 on both arms (kernel timeline + graph capture/breaks) to find what the skinny branch loses.

### Skinny c8 — root cause found and fixed (2026-11-26)
- Profile (torch profiler, n=4/8 decode): at n=8 skinny replaces 2520 TRT-LLM calls (7.3 → 4.1 µs, about −4 ms), but `act_and_mul` (2.5 → 4.5 µs) and `_gumbel_sample` (36 → 117 µs/step) get longer. So the end-to-end effect of the M=8 entries differs from their isolated kernel speedup.
- E19, paired c4/c8 only, vs `d6bccd2252` (2 GPU-swapped passes × 6 rounds, AOT 0):

| variant | c4 | c8 |
|---|---|---|
| skinny as-is (`0ed623973b`) | −4.32% (12/12) | **+2.87% (0/12)**, reproduced |
| PDL off in skinny FP8 | −0.11% | +3.32%: PDL helps, not the cause |
| **M=8 plan entries removed** | **−4.43% (12/12)** | **−0.25%** (pass 1 +0.97, pass 2 −1.46 → flat) |

- Fix: dropped the two M=8 entries ((1024,2048) and (512,2048)), added a comment explaining why, and amended the commit message. I also added the missing **Signed-off-by** (`0ed623973b` lacked it). New head **`b144c3ce70`** (author and committer = owner, no Co-authored-by). Ruff is clean.
- Skinny unit tests at the new head: 26 passed (2 fewer, because the tests are parametrized over the plan entries).
- Running now: full paired E2E c1–c32 at `b144c3ce70` vs `d6bccd2252` (E20, GPUs 0/1, with GSM8K per pass). PRREVIEW follows. M=8 now goes to TRT-LLM as on base, so the earlier GSM8K 8+8 (Δ −0.15 pp, noise) still applies to all other entries.

### Ack 6628a5d (skinny)
- GSM8K 8+8 at `b144c3ce70` vs `d6bccd2252` started: G11 h1 (A GPU2 / B GPU3) and h2 (A GPU5 / B GPU4) in parallel, 4 runs each, so each arm runs on two GPUs. E20 (full E2E c1–c32) continues on GPUs 0/1.
- The PRREVIEW body will state the M=8 finding in one or two sentences and use E20 for all concurrencies (including c4/c8). Opening waits for a maintainer review of #69949.

### Skinny `b144c3ce70`: E20 + G11 done (2026-11-26)
**E20, paired E2E vs `d6bccd2252`** (2 GPU-swapped passes × 6 rounds, AOT 0 in all 4 logs):

| c | pass 1 | pass 2 | mean | pairs faster |
|---|---|---|---|---|
| 1 | −9.77% | −8.06% | **−8.92%** | 12/12 |
| 2 | −9.09% | −8.40% | **−8.75%** | 12/12 |
| 4 | −5.08% | −3.71% | **−4.39%** | 12/12 |
| 8 | −0.21% | +0.18% | −0.02% | 5/12 |
| 16 | +0.18% | +0.76% | +0.47% | 2/12 |
| 32 | +0.73% | +0.15% | +0.44% | 3/12 |

**G11, GSM8K 8+8** (halves on GPU pairs 2/3 and 5/4): A 0.9625 (sd 0.0010), B 0.9629 (sd 0.0024), Δ +0.05 pp, Welch t +0.51. Flips A/B 16.7 vs floors 17.1 / 14.8; 0 systematic either way. Unit tests 26 passed.

- **c8 is fixed.** c16/c32 show a small, consistent-looking +0.45% (E18 measured them flat at −0.08 / −0.07). It is either GPU-pair noise or the remaining M=16 entry ((512,2048)) behaving like the M=8 ones. To settle it before the PRREVIEW: E21, c16/c32 only, `b144c3ce70` (GPUs 2/3) and `b144c3ce70` without the M=16 entry (GPUs 4/5), both vs `d6bccd2252`. If dropping M=16 removes the c16/c32 loss without costing c1–c4, I'll drop it too (new SHA → GSM8K 8+8 re-run per your rule).

### Ack d988a8b
- E21 already runs 2 GPU-swapped passes × 6 rounds per variant (pass 2 swaps A/B GPUs); AOT will be checked.
- Decision rule noted. E21 measures c16/c32 only. If M=16 is dropped, the new SHA gets a full E2E c1–c32 (which checks c1–c4 against E20 within 0.5 pp), plus unit tests, greedy single 32/32 + batched, and GSM8K 8+8.

### Skinny: correction. My `b144c3ce70` dropped TP2-only entries on TP1 evidence (2026-11-26)
**E21** (c16/c32, 2 GPU-swapped passes × 6 rounds, AOT 0):
- `b144c3ce70` as is: c16 +0.11%, c32 −0.11%;
- `b144c3ce70` without the M=16 entry: c16 −0.38%, c32 −0.08%.

**The mistake.** That M=16 entry is for (512, 2048), the **TP2** shared-expert gate_up_proj. With `shared_expert_intermediate_size` = 512, TP1 uses (1024, 2048). So:
- **E21:** the two variants run identical code at TP1. The 0.5 pp gap is pure noise (a useful noise calibration), and your decision rule does not apply.
- **`b144c3ce70`:** I had also dropped (512, 2048) **M=8**, which never runs at TP1. Only **(1024, 2048) M=8** caused the c8 regression.
- **Effect on TP2:** the old-plan TP2 run E13 had c8 −1.46% and c16 −4.05%, likely from exactly those two entries.

**Corrected head `07ec1f5c08`**: `0ed623973b` minus only the (1024, 2048) M=8 entry, a 2-line comment, and Signed-off-by (author and committer = owner; no Co-authored-by; ruff clean).
- At TP1 it runs exactly the same kernels as `b144c3ce70`, so E20 and G11 describe its TP1 behaviour. Per "the SHA that opens is the SHA that was measured", I am re-measuring anyway.
- **E22, running now, all on `07ec1f5c08` vs `d6bccd2252`:**
  - TP2 paired E2E c1–c32, GPUs 2,3 / 4,5;
  - TP1 paired E2E c1–c32, GPUs 0/1;
  - GSM8K 8+8 TP1, GPUs 6/7, halves swapped.
- Then: unit tests, greedy, PRREVIEW.
- Unless you object, I'll treat `07ec1f5c08` as the candidate.

**Side note for the PR (from the repo's `AGENTS.md`):** it asks for a commit attribution trailer (`Co-authored-by:` / `Assisted-by:`) and for the `pr-checklist` skill before opening. The owner rule is no Co-authored-by, so I've followed that. Please confirm whether the owner wants an `Assisted-by:`-style trailer on vLLM commits (this would apply to B1 #70028 too).

### Ack bc4fd92
- E22 TP1 and TP2 each use 2 GPU-swapped passes × 6 rounds (pass 2 swaps the A/B GPU sets); AOT will be checked. The PR body will carry both tables, and I will apply the TP2 rule (no concurrency beyond ±0.4% with most pairs slower; otherwise post before the PRREVIEW).
- Greedy at TP1 and TP2 (single 32/32 + batched) on `07ec1f5c08` after E22 frees the GPUs.
- Trailer: Signed-off-by only until the owner decides.

### Ack 32909e8 — skinny trailer amended (2026-11-26)
- New skinny head **`d5b0ff18ce9743fe8be18478238a3b1fb1570b2f`** = `07ec1f5c08` with `Assisted-by: Claude Code` directly above `Signed-off-by:` (message only; no Co-authored-by; author and committer = owner).
- Tree **`d8d98f9ab988e0160ebb89e26803d6fe6f6376dc`**, identical to `07ec1f5c08`'s tree (`git diff 07ec1f5c08 d5b0ff18ce` is empty), so E22 and the other measurements carry over.
- Unit tests on the same tree (base set + skinny file): base `d6bccd2252` 368 passed / 1 failed; skinny 396 passed / 1 failed. The 1 failure is the same pre-existing cuBLAS test on both sides.
- E22 (TP2 + TP1 E2E, G12 GSM8K) is still running. Greedy and the PRREVIEW come after it. B1 #70028 and MSA#53 stay unchanged.

### FAILURE (TP2 rule): E22 TP2 c8 is +2.44%, 0/12 pairs faster (2026-11-26)
E22, `07ec1f5c08` (tree = `d5b0ff18ce`) vs `d6bccd2252`, 2 GPU-swapped passes × 6 rounds, AOT 0 in all logs. Full tables: `results/b300/E22/`.

| c | TP2 mean Δ (pairs faster) | TP1 mean Δ (pairs faster) |
|---:|---:|---:|
| 1 | −1.75% (11/12) | −8.41% (12/12) |
| 2 | −8.84% (12/12) | −8.80% (12/12) |
| 4 | −4.30% (12/12) | −4.59% (12/12) |
| 8 | **+2.44% (0/12)** | −0.06% (8/12) |
| 16 | −3.91% (12/12) | −0.09% (6/12) |
| 32 | −0.18% (9/12) | −0.16% (8/12) |

- **TP1** is clean: c1–c4 match E20, and c8–c32 are inside ±0.4%.
- **GSM8K G12** (TP1, 8+8, halves GPU-swapped): base 0.9622, skinny 0.9624, Δ +0.02 pp (t +0.17). Flips are 16.8 per base/skinny pair against floors of 17.7 (base/base) and 15.7 (skinny/skinny). No question is systematically right on one side and wrong on the other.
- **TP2 c8 fails the rule.** At batch 8 the only TP2 entry that runs is **(512, 2048) M=8**, the TP2 shared-expert gate_up. That is the same shape role as the (1024, 2048) M=8 entry I removed for TP1: it wins in isolation (1.70×) but loses end to end. TP2 c16 (where (512, 2048) M=16 runs) is −3.91%, 12/12.
- **E23 is running now** (TP2 only, c4/c8/c16, 2 passes × 6 rounds), using the variant `d5b0ff18ce` minus the (512, 2048) M=8 entry:
  - (a) base `d6bccd2252` vs variant, GPUs 2,3 / 4,5;
  - (b) `d5b0ff18ce` vs variant, GPUs 0,1 / 6,7.
- **Proposal.** If (a) shows c8 inside ±0.4% and c4/c16 unchanged, drop that entry with a comment like the TP1 one, re-run TP2 c1–c32, and only then write the PRREVIEW. No PRREVIEW until you reply.

### Ack 2b0e393
- Understood: `d5b0ff18ce` is not a candidate. If E23(a) clears c8 (inside ±0.4%, with c4/c16 within 0.5 pp of E22), I'll drop the (512, 2048) M=8 entry with a matching comment.
- Then on the new SHA: TP2 and TP1 E2E c1–c32 (TP1 in short form); greedy single 32/32 at TP1 and TP2; unit tests. GSM8K is carried over from G12, with the reason stated in the PRREVIEW.
- If E23(a) does not clear c8, I'll post it before changing anything else.

### E23 clears TP2 c8 → new skinny head `fd7eea8834` (2026-11-26)
E23 ran TP2 only, at c4/c8/c16, with 2 GPU-swapped passes × 6 rounds and AOT 0. The variant is `d5b0ff18ce` minus the (512, 2048) M=8 entry. Tables are in `results/b300/E23/`.

| c | (a) base → variant | E22 base → `d5b0ff18ce` | (b) `d5b0ff18ce` → variant |
|---:|---:|---:|---:|
| 4 | −4.35% (12/12) | −4.30% | +0.21% (5/12; pass 1 +0.74%, pass 2 −0.33%: GPU-pair effect) |
| 8 | **−0.08%** (7/12) | +2.44% | **−2.50% (12/12)** |
| 16 | −4.16% (12/12) | −3.91% | +0.28% (3/12) |

The criteria are met: c8 is inside ±0.4%, and c4/c16 are within 0.5 pp of E22. (b) confirms that this one entry caused the c8 regression.

**New head `fd7eea88349237f939b72052c61b2fb6f1045217`** (tree `6dd669471d`):
- It is `d5b0ff18ce` minus the (512, 2048) M=8 entry.
- The plan comment now covers both shapes: "The shared-expert gate_up_proj shapes (1024, 2048) and (512, 2048) have no 8-token entry: they win in isolation (1.38x / 1.70x), but decode at batch 8 is 2-3% slower end to end with them (TP1 / TP2)." The matching commit-message paragraph is updated too.
- Trailers: `Assisted-by: Claude Code` + `Signed-off-by:`. Author and committer are the owner. ruff is clean.
- `git diff d5b0ff18ce fd7eea8834` touches only that table line and the comment, so G12 GSM8K carries over.

**E24, running now** on `fd7eea8834` vs `d6bccd2252`:
- TP2 E2E c1–c32 (GPUs 2,3 / 4,5);
- TP1 E2E c1–c32 (0 / 1);
- unit tests (6 / 7, with full output so the failing test is named);
- then greedy TP1 (6 / 7) and greedy TP2 (0,1 / 6,7).

PRREVIEW after that.

### E24 done; skinny PRREVIEW posted (2026-11-26)
**Skinny E24**, on `fd7eea8834` vs `d6bccd2252`: 2 GPU-swapped passes × 6 rounds, AOT 0 in every server and greedy log.

| c | TP2 Δ (pairs faster) | TP1 Δ (pairs faster) |
|---:|---:|---:|
| 1 | −3.48% (10/12) | −9.03% (12/12) |
| 2 | −7.89% (11/12) | −8.75% (12/12) |
| 4 | −4.50% (12/12) | −4.44% (12/12) |
| 8 | −0.17% (7/12) | −0.15% (7/12) |
| 16 | −3.78% (12/12) | +0.25% (3/12) |
| 32 | +0.28% (6/12) | +0.16% (6/12) |

- **TP2 rule:** met. No concurrency is slower beyond ±0.4%.
- **TP2 c1 is noisy:** pass 1 −0.27%, pass 2 −6.69%.
- **Unit tests (full output this time):**
  - PR: 394 passed, 1 failed. Base: 368 passed, 1 failed.
  - The failure is the same on both sides: `test_w8a8_block_fp8_torch_scaled_mm_matmul`, cuBLASLt `NOT_SUPPORTED`.
  - The skinny file alone: 26 passed in 30.45 s.
- **pre-commit:** all hooks pass on the 5 changed files.
- **Greedy:**
  - TP1: single 3/32, batched 200/256 (floors 32/32, 162/256).
  - TP2: single 8/32, batched 222/256 (floors 32/32, 253/256).
- **GSM8K:** carried over from G12, per 2b0e393.

**PRREVIEW:** `results/b300/PRREVIEW/skinny_on_69949/` (SUMMARY.md, title.txt, body.md, diffstat.txt, raw_*).
- Head `fd7eea88349237f939b72052c61b2fb6f1045217`, on #69949 `d6bccd2252`.
- The body follows the repo PR template and the pr-checklist skill, plus `## AI assistance` and the footer.
- SUMMARY.md lists 5 points for you: TP2 c1 noise, complexity band, CI gap on L4, sync check not run, and the dropped template last line.
- Not opening until #69949 has a maintainer review (still none) and you post `APPROVED TO OPEN: skinny @ fd7eea8834…`.

**GPUs:** the user asked for 1–2 GPUs. GPUs 6 and 7 have been left free since 07:12 and are now in use by the user. My jobs run only on 0–5.

### Ack 493ff0b
- Holding `fd7eea8834` as is: no rebase, no amend. Not opening until #69949 has a maintainer review and you post `APPROVED TO OPEN`.
- Sync check skipped. I'll wait for the owner's answer on the AI-section wording and change `body.md` only once it arrives.

### Ack 1a95f99: B9 started (2026-11-26)
**Arms.** B8 / B1 trees, each with its own build:
- A = main `fb159244e6`;
- B = v1 `c5ef00f633`;
- C = v2 `152bca1e65`. v2's csrc change has its own build; it is the same tree that served as the B1 base.

**Eval.** `gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --max-concurrency 64 --temperature 0.6 --top-p 0.95 --seed S`, with a per-question dump.
- 8 seeds (101–108). Every arm runs the same seed set, so runs pair up by seed across arms.
- Run order rotates per seed.

**GPU layout.** Two halves run concurrently, with the arms rotated across GPUs:
- X: seeds 101–104 on GPUs A0 / B1 / C2;
- Y: seeds 105–108 on GPUs A4 / B5 / C3.

Each arm gets 4 runs on each of two GPUs. GPUs 0–5 only. AOT will be checked.

**Report.** Mean, sd and Welch t for v1 and v2 vs main; per-question counts for questions differing by ≥ 4 of 8; and a paste-ready setup paragraph.

### B9 done: v1 is within noise of main under sampling (2026-11-26)
Full write-up, a paste-ready paragraph and the script are in `results/b300/B9/`. Sampled GSM8K (t=0.6, top_p 0.95), 8 shared seeds per arm, GPU-rotated, AOT 0, 0 invalid.

| arm | mean (sd) | Δ vs main, Welch t | Q's with correct counts differing by ≥ 4/8 (null mean / p95) |
|---|---|---|---|
| main `fb159244e6` | 0.9594 (0.0027) | | |
| v1 `c5ef00f633` | 0.9581 (0.0033) | −0.13 pp, t −0.87 | 0 (2.4 / 5) |
| v2 `152bca1e65` | 0.9567 (0.0024) | −0.27 pp, t −2.13 | 1, better in v2 (3.0 / 7) |

- **Noise scale.** v2 computes the same numbers as main, so its −0.27 pp (t −2.13) is the noise scale under sampling. Batch composition at concurrency 64 changes trajectories: same-seed output lengths match on only about half the questions.
- **v1.** v1's gap is half of v2's, and no question shifts.
- **Implication.** By your stated rule (drop the second commit if v1 is within noise), the second commit can be dropped. That is the owner's call.
- **Caveat.** A 0.27 pp gap with |t| about 2 between equal-numerics arms means 8 runs can't resolve differences below about 0.3 pp. Say if you want 8 more seeds; that takes about 15 min on GPUs 0–5.

### Skinny opened per aafa417: vllm-project/vllm#70205 (draft)
https://github.com/vllm-project/vllm/pull/70205. Head `fd7eea88349237f939b72052c61b2fb6f1045217`, base `main`, draft, no labels or reviewers. Details are in `PRREVIEW/skinny_on_69949/opened.md`, and the posted body is in `body.final.md`.

### B10 ack: #69778 smoke check at `e1df7ec5f6` (running)
Fetched read-only; no push. Base = merge base `29fa74e654`. Running on GPUs 0–1 (pytest used GPU 2). Fusion counter and pytest are done; the paired TPOT run at c1/c4/c8 (1 pass × 4 rounds) is in progress. One paragraph follows when it finishes.

### B10 done: #69778 at `e1df7ec5f6` passes the smoke check
On the new head, real weights, TP1, the fused SiLU+group-128 producer runs on every shared-expert down_proj `(2048, 512)` (640/640 calls, 40 layers × 16 steps); on base `29fa74e654` the key is None and nothing fuses. So the `create_weights` move keeps the fusion live. Paired median TPOT (1 pass × 4 rounds, AOT 0): c1 −1.99% (4/4), c4 −0.36% (3/4), c8 −1.40% (4/4). This is in line with earlier runs. The two requested test files give 56 passed. The head carries v1 only (`c5ef00f633`; the bit-exact second commit was dropped), so server greedy matches on 1/4 prompts (first divergence at tokens 7/40/50). That matches v1's known non-bit-exact quant and B9's "v1 within noise". Details are in `results/b300/B10/`. No GitHub actions; GPUs are free.

### B11 ack: SM103 perf hunt (FA4 / QuACK / decode profile)
Starting on main `97dfd07879`, GPUs 0–5. A re-test of #70028 / #70205 takes priority if one arrives. Findings go to `results/b300/B11/FINDINGS.md`; no GitHub activity.

### B11 done: SM103 perf hunt. Details in `results/b300/B11/FINDINGS.md`
On main `97dfd07879`, Qwen3.6-35B-A3B-FP8 TP1 (real weights) and Qwen3.8-Flash-Next-FP8 TP1 (dummy weights; the download is too slow).

Ranked candidates:
1. **Sampler `top_k_max`, i.e. open #64116.** Measured E2E on main+#64116: c1 −3.62%, c8 −3.21%, c32 −1.57% (4/4 pairs each). It is the cheapest win, and the code already exists upstream.
2. **Shared-expert gate tail.** Gate GEMV + sigmoid + mul + add is 7.6 µs per layer; a fused prototype takes 1.8 µs, which would save ~7% / 6% / 5% of the step at M=1/8/32. It overlaps #43190 (stale) and #60846 (SM120 opt-in); neither fuses the add.
3. **Batch-1 MoE GEMMs at 17–21% of roofline** (FC1 403 vs 84 µs, FC2 241 vs 42 µs), even though FlashInfer autotunes them. Worth ~10% at c1, but it's a large change and no open PR covers it.
4. **The post-attention norm output is quantized twice** (MoE and shared expert): ~3% at M=1. Not covered.

Other results:
- **FA4.** Qwen3.6 runs FlashInfer by default on SM10x, by design; forcing FA4 is 1–2.6% slower. Qwen3.8 uses its own QSA attention, so FA doesn't apply. The FA pin is fine: it's a merged squash commit on the fork's main.
- **QuACK.** Its norms are 5–43% slower than the FlashInfer CuTe norms vLLM already uses, so no case there. Its softmax is 4–11× faster than torch on a 248k vocab, which only matters for full-vocab softmax paths.
- **Already covered or near roofline.** Dense FP8 CUTLASS at 13% of roofline is covered by #69949 / #70205. GDN recurrent is at 77% of roofline.

No PR built and no GitHub activity. GPUs are free.

## B12 — ack 2026-11-28
- Fetched `b8b5ff3926` read-only (parent `269fca48a1`); `git patch-id --stable` equals the B1 commit `e77ca1439c` tested earlier.
- Running on GPUs 0–3: fusion count (real weights, TP1) and pytest (B1 test file + `test_quant_activation_contract.py`) on both trees; then paired TPOT c1/c4/c8/c32 (1 pass × 4 rounds) + GSM8K 1+1 on GPUs 0–1. ETA ~1.5 h.

## B12 — done 2026-11-28 (details: results/b300/B12/)
#70028 at `b8b5ff3926` (patch-id = B1 `e77ca1439c`) vs parent `269fca48a1`, Qwen3.6-35B-A3B-FP8 TP1, real weights, B300. **pytest** (B1 test file + `test_quant_activation_contract.py`): base 108 passed, PR 144 passed (+36 new), 0 failed. **Fusion fires**: `gemma_add_rms_norm_quant` with `kFp8Dynamic128Sym` 640× (40 layers × 16 steps); CUTLASS block-FP8 gets pre-quantized input for GDN `in_proj_qkvz` (12288, 2048) 480× and attention `qkv_proj` (9216, 2048) 160×, on top of #69778's `down_proj` 640× (base has only the latter). **Paired median TPOT** (1 pass × 4 rounds, AOT load 0): c1 2.891 → 2.761 ms (−4.49%, 4/4), c4 −2.68% (4/4), c8 −2.65% (4/4), c32 −0.33% (3/4). **GSM8K 1+1**: base 0.9583, PR 0.9636, 23 per-question flips (within the B1 noise of ~17 base/base). Greedy 3/4 prompts identical, 1 diverges at token 7 (norm reduction order, as in B1). No GitHub actions.

## B13 — ack 2026-11-28 ~07:10Z
- Missed the B13 push for a few hours: this node lost its GitHub connection (fetch timeouts in my poller log from ~03:00Z); it is back now. B12 results are at b538dd8.
- Base worktree at `14e9c30d86`, quack-kernels 0.6.5. Downloading Qwen3-8B and Qwen3-30B-A3B (BF16; not cached here). Writing P1/P2/P3 locally from the description; kernel microbenches first, then paired E2E. Code stays local.

## B13 — progress 2026-11-28 (kernel microbenches done; E2E running)
P1/P2/P3 rebuilt locally from the description on `14e9c30d86` + quack-kernels 0.6.5 (code local only). All three run on SM103 as described; the pinned config (128x256, cluster 2x1, dynamic persistent) is valid here, no workaround needed. Cold L2, CUDA graph, median of 7 rounds (`results/b300/B13/mb_p*.json`):
- **P1** (E=128, k=8) vs `ops.topk_softmax`: ids identical at every M, weights within 1.2e-7. M=1 3.52 → 1.84 µs (1.91x); M 8–256 1.45–1.66x; M=1024 2.09x; M=8192 3.70x.
- **P2** (Qwen3-30B-A3B shapes) vs `fused_experts` (Triton, no B300 tuned config file): M=64 1.06x, 256 1.08x, 1024 1.14x, 2048 1.30x, 4096 1.45x, 8192 1.58x. Rel. error vs FP32 reference: QuACK 0.0029, Triton 0.0043.
- **P3** (Qwen3-8B gate_up, K=4096, N=2x12288) vs `F.linear` + `silu_and_mul`: M=1/8 (F.linear + 2-D Triton SiLU-mul) 1.05x; M=16/32 1.06x; M=128 1.20x; M=512 1.23x; M=2048 (untuned) 1.15x. Rel. error vs FP32: QuACK 0.0017, cuBLAS+C++ 0.0035.
Paired E2E (2 passes x 6 rounds, GPU-swapped, GSM8K once per arm) running now, all three at once on GPU pairs 0/1 (P1), 2/3 (P2), 4/5 (P3, after pre-tuning M=16..512 step 8).

## B13 — done 2026-11-28 08:30Z (details: results/b300/B13/README.md)
All three prototypes run on SM103 without workarounds; in-server path counts confirm each fires. Paired E2E (2 passes x 6 rounds, GPU-swapped): **P1** TPOT c1 -3.18% (12/12), c8 -2.15% (6/12), c32 -1.32% (9/12), c64 -1.08% (8/12). **P2** prefill TTFT c1 -8.62%, c4 -12.89%, c16 -15.11% (all 12/12). **P3** TPOT c1 -0.11%, c8 -0.06% (flat), c32 +0.52% (1/12), c128 -2.75% (12/12). GSM8K once per arm: P1 0.9310 -> 0.9280, P2 0.9280 -> 0.9212 (-0.68 pt, 15/6 flips; base/base gap here is 14 questions, so one run cannot settle it — a few more runs would), P3 0.9022 -> 0.9052. Note: P1 skips the MoE padding mask that base applies. No GitHub actions.

## B14 — ack 2026-11-28 08:35Z
Running: (1) default MoE backend vs `--moe-backend triton` + P1 + P2 on GPUs 0/1, TTFT (4096/1, c1/c4/c16) then TPOT (c1/c8/c32/c128), 2 x 6, GPU-swapped; (2) GSM8K 4 runs per arm, Triton vs P2, both servers up at once on GPUs 2/3 (2 runs, swap, 2 runs). Optional profile after, on a free GPU.

## B14 — partial 2026-11-28 15:30Z (sorry for the silence: item 1 finished at 09:45Z, but my GSM8K-repeat script hung and I did not report)
**Hang, cause:** the GSM8K repeat loop used a bare `wait`, which also waited on the two background vLLM servers, so it blocked forever after run 1 (08:24Z). Killed at 15:20Z, fixed (`wait` on the GSM PIDs only), and relaunched: 4 runs per arm, ~1 min per run, results within ~15 min.

**1. Default MoE (SM103 picks `TrtLlmBf16ExpertsMonolithic`, confirmed in the log) vs `--moe-backend triton` (`TritonExperts`) + P1 + P2.** Qwen3-30B-A3B BF16, 2 passes x 6 rounds, GPU-swapped; + means the QuACK arm is slower.
| | c | pass 1 | pass 2 | mean | QuACK arm faster |
|---|---:|---|---|---:|---:|
| TTFT 4096/1 | 1 | 51.68 -> 48.90 ms (-5.38%) | 50.19 -> 53.93 (+7.45%) | +1.03% | 6/12 |
| | 4 | 132.6 -> 142.2 (+7.21%) | 130.2 -> 140.3 (+7.80%) | **+7.50%** | 0/12 |
| | 16 | 481.9 -> 505.5 (+4.89%) | 476.4 -> 513.6 (+7.81%) | **+6.35%** | 0/12 |
| TPOT 512/512 | 1 | 2.862 -> 3.234 (+12.98%) | 2.837 -> 3.259 (+14.88%) | **+13.93%** | 0/12 |
| | 8 | 4.950 -> 5.669 (+14.52%) | 4.797 -> 5.601 (+16.75%) | **+15.64%** | 0/12 |
| | 32 | 7.219 -> 8.501 (+17.76%) | 7.440 -> 8.454 (+13.63%) | **+15.70%** | 0/12 |
| | 128 | 11.34 -> 13.07 (+15.24%) | 11.38 -> 13.08 (+14.94%) | **+15.09%** | 0/12 |
Same picture as B200: the default TRT-LLM path beats Triton + P1 + P2. GSM8K (once per arm, from the TPOT run): default 0.9249, QuACK arm 0.9265.

**Optional profile** (`vllm bench forward`, one 4096-token prefill, real weights/routing; `results/b300/B14/prof_breakdown.txt`): forward 37.8 ms (default) vs 41.0 ms (QuACK arm). Per category, default vs QuACK, µs: MoE FC1+SwiGLU 10814 vs **9399** (QuACK 1.15x faster); MoE FC2 **5063** vs 5598 (TRT-LLM 1.11x faster); routing + permute + combine **2218** vs 4724 (QuACK arm pays `expandInputRows` 2267 + `finalizeMoeRouting` 1553 + cub sort ~620; TRT-LLM gathers inside FC1 and only runs finalize 1489 + 2 small routing kernels); attention / dense / norms equal. So QuACK's grouped GEMM wins FC1, but the separate permute/unpermute costs more than the GEMM gains.

**2. GSM8K x4 per arm (Triton vs P2):** running now on GPUs 2/3.

## B14 — done 2026-11-28 15:40Z
**2. P2 GSM8K, 4 runs per arm** (Qwen3-30B-A3B, `--moe-backend triton`, both servers up at once, 2 runs then GPUs swapped, 2 runs; `results/b300/B14/gsm_repeats.txt`):
- Triton: 0.9249 / 0.9242 / 0.9227 / 0.9257, mean 0.9244 (sd 0.0013). P2: 0.9227 / 0.9227 / 0.9242 / 0.9242, mean 0.9234 (sd 0.0009). Δ −0.10 pt.
- Per-question flips: within Triton 13.5 per run pair (11–16), within P2 18.0 (16–20), between arms 18.2 (14–22). One question is right in all 4 Triton runs and wrong in all 4 P2 runs; none the other way.
- So the B13 −0.68 pt was mostly noise: the between-arm flips match P2's own run-to-run flips. P2 is somewhat less run-to-run stable than Triton (18 vs 13.5 flips within arm), but its mean accuracy is within one sd.
Item 1 and the profile: see the partial above. B14 complete; GPUs free.

## B15 — ack 2026-11-28 15:50Z
Starting NVFP4 kernel scouting on SM103: pick an ungated NVFP4 dense checkpoint, serve-profile vLLM's linear / act-quant kernels, check QuACK consumes the weights as is, then microbench plain and gated NVFP4 GEMMs vs vLLM's. Kernels only, code local.

## B15 — done 2026-11-28 (phase 1; details: results/b300/B15/SUMMARY.md)
nvidia/Qwen3-32B-NVFP4 on SM103: every linear runs FlashInfer **cute-dsl** NVFP4 GEMM (57-59% of GPU time at c1/c32/prefill), with fused silu_mul+fp4 quant and Inductor-fused norm+fp4 quant. **QuACK reads vLLM's NVFP4 weights/scales as is** (same swizzled SF layout, zero-copy view) and its plain GEMM output is bit-identical to cute-dsl. It is not faster in general: 0.84-1.0x on gate_up, 0.41-0.64x on down_proj at M <= 32, wins only at isolated points (o_proj M 32-128 1.10-1.27x, qkv M=128 1.30x). **One GO candidate:** gate_up gated GEMM (SwiGLU in the epilogue, bf16 postact) + fp4 quant beats vLLM's GEMM + fused silu_mul_fp4 by **1.18-1.24x at M >= 512** (prefill; ~41% of prefill GPU time), but is 0.83-0.91x at M <= 32. It needs a one-time interleave of the gate_up weight rows. **Blocked:** QuACK's gated GEMM with NVFP4-quantized postact does not compile on SM10x in 0.6.5 (all 48 configs fail an epilogue smem-layout assert). No GitHub actions; code local.

## B15b + B16 — ack 2026-11-28 16:20Z
Doing B15b first (stock `swiglu_quant_mod` with NVFP4 output, configs pinned, no tile_n=192; my B15 attempt was an alpha-carrying copy with autotuned configs), then B16.

## B15b — done 2026-11-28 (details: results/b300/B15/SUMMARY.md, "B15b" section; data b15b.jsonl)
**Correction to B15:** the gated GEMM with NVFP4-quantized postact does work. My B15 failure came from autotuning; pinned configs compile and are correct (252 runs, 0 wrong, no tile_n 192). gate_up + SwiGLU + NVFP4 out as one QuACK kernel (stock `swiglu_quant_mod`, plus an alpha copy for NVFP4's pre-SiLU scale) vs cute-dsl GEMM + fused `silu_mul_cvt_fp16_to_fp4`, on Qwen3-32B-NVFP4 gate_up (CUPTI, cold L2): **M=512 1.27x, 2048 1.39x, 8192 1.29x**; M=128 1.03x; M <= 32 0.80x. fp4 codes 98.7-98.9% and SF bytes 96.6-100% identical to vLLM (the rest are 1-step rounding differences: fp32 vs bf16 before SiLU); error vs FP32 is the same. **down_proj (cute-dsl) consumes QuACK's output as is**, with the same error. Prefill-only GO; needs M dispatch + pinned configs + a one-time gate_up row interleave. Next: B16.

## B16 — done 2026-11-28 (details: results/b300/B16/SUMMARY.md)
Vocab 151936, top_k 20 / top_p 0.95. The full-vocab softmax is FlashInfer's internal `torch.softmax` in `top_k_top_p_sampling_from_logits` (vLLM passes a per-row k with no `top_k_max`, so the fast path is skipped). `cunn_SoftMaxForward` uses one block per row, about 49 µs at B=1. **QuACK softmax is 6-13x faster at B <= 16 (fp32), but has two bugs:** NaN on top-k-masked logits (any fill <= -1e10, including -inf), and all-NaN for bf16/fp16 input at vocab ~151-152K. With a -1e6 finite-fill workaround the chain is 1.51 / 1.42 / 1.36x at B = 1 / 4 / 16 (0.85x at 256), and seeded top-p tokens are 100% identical. **FlashInfer's `top_k_max` fast path already beats that from B=4 (1.49-2.94x)**, and #64116 / #65649 target it. QuACK top-k needs N a power of two <= 4096, so it is unusable at this vocab; no temperature fusion without QuACK changes. Recommendation: no QuACK sampler candidate. Overlapping open PRs listed.

## B17 — ack + progress 2026-11-28 16:45Z
Prototype built (local): `Qwen3MLP` on NVFP4 calls one custom op. For M >= 128 it runs QuACK gated GEMM + alpha + SwiGLU + NVFP4 out (pinned configs: 128x128 c1 below M 512, 256x256 c2 from 512; never tile_n 192). Below 128 it runs vLLM's cute-dsl GEMM + `silu_and_mul_nvfp4_quant` (same kernels as today). The output codes plus swizzled SF are passed to down_proj as a `QuantizedActivation`. The output global scale is down_proj's `input_global_scale_inv`, and the SF goes straight into vLLM's (Mpad128, I/16) swizzled buffer from `create_fp4_output_tensors`. Op check: codes 98.9% and SF 98.6% identical to vLLM's branch at M = 128/300/512/2048/4096. Model check: the path fires 64x per prefill (eager) and works under torch.compile + CUDA graphs. **Memory: the interleaved gate_up copy costs 8.85 GiB** (KV cache 204.84 -> 195.99 GiB). Greedy, 4 prompts x 48 tokens: 2/4 identical, 2 diverge at tokens 3 / 31. Paired E2E (TTFT GPUs 0/1, TPOT GPUs 2/3) and GSM8K 4 runs per arm (GPUs 4/5) are running.

## B17 — done 2026-11-28 17:35Z (details: results/b300/B17/SUMMARY.md)
Qwen3-32B-NVFP4, gate_up as one QuACK kernel (gated GEMM + alpha + SwiGLU + NVFP4 out) for M >= 128. Its output feeds vLLM's cute-dsl down_proj as is (same swizzled SF buffer; output global scale = down_proj `input_global_scale_inv`). Paired E2E 2 x 6, GPU-swapped: **prefill TTFT c1 -9.63 %, c4 -9.82 %, c16 -11.40 % (12/12 each); TPOT c8 -0.24 %, c32 -0.33 % (flat, below the threshold), c128 -5.63 % (12/12).** GSM8K x4: 0.9399 -> 0.9416. B17 is less run-to-run stable (35 within-arm flips vs 0.5): the kernel is bit-deterministic, but numerics switch with batch M (two branches differ in ~1 % of fp4 codes). Memory: +8.85 GiB for the interleaved gate_up copy. Warm-L2 4-layer check confirms the M = 128 cut (1.15x; 64: 1.08x; 32: 0.88x). Landing needs are listed for QuACK (alpha on gated_quant_mod, safe autotune, blockscaled concat_layout) and vLLM (NVFP4 gated-producer hook, interleave at load, M dispatch, batch-invariance note).

## B18 — ack 2026-11-28
- Read B18 and W23/W24. Rebuilding the MXFP8 MLP prototype (FlashInfer mxfp8_quantize → QuACK swiglu_quant_mod MXFP8 out → QuACK MXFP8 down GEMM, M ≥ 128) on SM103, Qwen3-8B-FP8 TP1. First: check SM103 UE8M0 weight-scale rounding and the in-server quant path, then a warm kernel sweep, then paired E2E (2 x 6, GPU-swapped) + GSM8K once per arm. Code stays local.
