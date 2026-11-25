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
