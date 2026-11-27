# FA4 (CuTe DSL) at vLLM's pin vs. upstream Dao-AILab main, for Qwen3.6/3.8 hd256 on B300

Date: 2026-11-27. This is read-only research. No GPU jobs were run and nothing was written to GitHub.
Clones: `<scratch>/fa`. Its remotes are `origin` (Dao-AILab) and `vllm` (vllm-project fork).
Pinned tree: a local scratch clone.

| ref | SHA | date |
|---|---|---|
| vLLM checkout | 97dfd07879 (`~/kt/b11_main`) | |
| vLLM pin (`cmake/external_projects/vllm_flash_attn.cmake:42`) | **1b32c5ca** (vllm-project/flash-attention) | 2026-11-18 |
| fork main HEAD | 8bfb5666 | 2026-11-24 |
| upstream Dao-AILab main HEAD | 2676a7be | 2026-11-25 |
| upstream base of the last fork sync | 54b044c8 (via fork #213 = af5422c5) | 2026-11-01 |

## Summary

- **The pin is a merged commit, not a PR head.**
  - 1b32c5ca is the squash-merge of fork PR #222, which merged 2026-11-18T20:22Z. It is on fork `main`.
  - Its tree is byte-identical to the PR head 08a4a410 (both have tree b02387e4).
  - Fork main has exactly one commit that the pin lacks: 8bfb5666, fork #206, "[Cute,Fwd,Sm90] Native FP8 forward for Hopper". It is SM90-only and irrelevant on B300.
- **The vendored FA4 code matches the pin.**
  - `vllm/vllm_flash_attn/cute/` is identical to `flash_attn/cute` at 1b32c5ca once the `vllm.vllm_flash_attn` → `flash_attn` import rewrite is undone.
  - Only packaging files differ (LICENSE, README, pyproject, etc.).
- **The pin's FA4 = upstream 54b044c8 + fork-only additions.**
  - The fork-only additions are: SM90 FP8 and hd512, the MLA sparse decode kernel, fused FP8 output quant (`output_quant_key`), `dynamic_causal`, and SM90 split-KV.
  - Upstream has **34 commits touching `flash_attn/cute`** since 54b044c8, out of 45 total.
  - None of them adds a new inference feature for BF16 hd256 on SM100/SM103.
  - Two of them change hd256/GQA *forward selection* in ways that matter for us (c055bdc6 #3148 and 8e3cb052 #3087).
- **Most important caveat: on B300, vLLM does not use FA4 for these models by default.**
  - `vllm/platforms/cuda.py:158-165` puts FLASHINFER ahead of FLASH_ATTN on SM10x for causal attention.
  - Our own logs confirm it. `~/b300-work/B1_profile/serve.log` and `B8/check_base.log` show `attention | 10 | FLASHINFER`.
  - Everything below therefore applies only to runs with `--attention-backend FLASH_ATTN` (or wherever FlashInfer rejects a config).

## Q1. The pin, fork PR #222, and fork main

- **PR #222** "[Build] Disable append-KV kernels in FA2 too (follow-up to #107)" is MERGED.
  - Merged 2026-11-18T20:22:14Z.
  - Head 08a4a4105e (on chatouis' fork, one commit on top of 8dd9eaea).
  - Merge commit **1b32c5caf7**, whose parent is 8dd9eaea.
- **The pin is not the PR head.** It is the squash-merge commit, and it is reachable from `vllm/main`. `git branch -r --contains` lists `vllm/main`.
- **History of the pin in vLLM PR #68348.**
  - The PR first pointed at `chatouis/flash-attention@08a4a410`.
  - Commit 82651c2f48 (by M. Bonanni) switched it to `vllm-project/flash-attention@1b32c5ca` before merge.
  - The merged vLLM commit is d9162cc8fc, dated 2026-11-19. Its title still carries "[Do Not Merge]"; the GitHub title was later edited.
  - The content is fine: identical tree to the tested head, and it is on the fork's main.
- **Fork main commits not in the pin:** only **8bfb5666** (#206, 2026-11-24).
  - It touches `config.py`, `flash_fwd.py`, `flash_fwd_sm90.py`, `interface.py`, `softmax.py`, `tile_scheduler.py` and `utils.py` (+1886/-121).
  - All of it is gated on SM90 `native_fp8`. No effect on SM100/SM103.
- **Conclusion:** building from 1b32c5ca does not matter in practice. There is no divergence from fork main other than one Hopper-only commit.
- **What the PR changes:** #222 itself removes FA2 append-KV instantiations (FA2 only). Relative to the previous pin 8dd9eaea, FA4 is unchanged.

## Q2. Pinned FA4 vs. upstream main

### Sync base

- vLLM #65203 (merged 2026-11-09) consumed fork #213 = af5422c5.
- #213 is a three-way merge of upstream ce088ab9..**54b044c8** (upstream 2026-11-01, "#2743 explicit forward config autotuning").
- 54b044c8 is not a git ancestor of the pin; it was imported as a merge and squashed.
- `git diff 54b044c8 1b32c5ca -- flash_attn/cute` shows only fork additions: 16 files, +3607/-267. Most of it is in `flash_fwd_sm90.py`, `flash_fwd_mla_decode_sm100.py` and `interface.py`.
- `flash_fwd_sm100.py` differs by only 18 lines (fused FP8 output scale and the `mDynamicCausal` arg).
- No upstream SM100 forward code appears to have been dropped. I checked the SM100 forward, config, tile_scheduler and blackwell_helpers diffs by hand.

### Upstream `flash_attn/cute` commits after 54b044c8 (34), by topic

Sizes are lines +/- in `flash_attn/cute` only.

**Hits the hd256 / GQA forward selection that our models use**

| commit | PR | what | size |
|---|---|---|---|
| c055bdc6 | #3148 | hd256: keep PackGQA when 2CTA is impossible (page_size ∉ {None,128} or 2CTA disabled). | +7/-1, config.py |
| 8e3cb052 | #3087 | Route padded or many-slot SM100 varlen forward to `DynamicPersistentVarlenScheduler` (only real tiles launched). A routed call is re-selected with PackGQA for hd256. | +134/-8 |
| f4cc3b0b | #3086 | Dynamic varlen scheduler binds tiles late and self-resets its semaphore. This is the prerequisite for #3087. | +130/-39 |
| bee0f085 | #3017 | Softmax stats released per stage by default. The cross-stage table has no hd256 entry, so hd256 BF16 switches to same-stage. Output is bitwise identical, but hd256 perf was not reported. | +103/-19 |

**FP8 / MXFP8 attention (SM100)**

| commit | PR | what | size |
|---|---|---|---|
| dba62ef0 | #3018 | MXFP8 quantization and scale-layout helpers (`blockscaled_utils.py`). | +130 |
| 9985a28e | #3027 | **API break:** `q/k/v_descale` → `scales=TensorwiseScales(...)`. | +115/-90 |
| 69384e9c | #3019 | MXFP8 QK with BF16 PV forward. Dense, unsplit, hd64/128 only; varlen and SwapAB reject it. | +541/-108 |
| 69832334 | #3020 | Non-causal native MXFP8 PV. | +468/-98 |
| cc7b69d4 | #3022 | Causal native MXFP8 PV with a BF16 band. | +267/-65 |

**SM107/SM120/SM80-only (not B300)**

| commit | PR | what | size |
|---|---|---|---|
| 7c53542e | #3029 | sm_107 15-bit SMEM descriptor address (sm_100/103 unchanged). | +22/-7 |
| b2edfab9 | #3121 | Compile autotune candidates for sm_107a. | +7/-7 |
| ed8ea723 | #2998 | SM120 BF16/FP16 tiles. | +10/-4 |
| a47a0883 | #2954 | SM80 local-window skip. | +1/-1 |
| b0cc5819 | #2752 | `learnable_sink` on SM80/SM120. SM100 already had it. | +181/-105 |

**MLA (DeepSeek-style), not our models**

| commit | PR | what | size |
|---|---|---|---|
| 1dc8232f | #3091 | Expose MLA CTA choice. | +20/-6 |
| 5f7b05a7 | #3118 | Pad dense 2CTA MQA heads. | +29/-13 |
| 2ed4a1f0 | #3140 | Pad dense 2CTA MQA heads (follow-up). | +27/-12 |

**Robustness and edge-case fixes in the forward/interface** (do not affect normal shapes)

| commit | PR | what | size |
|---|---|---|---|
| 8fa1e462 | #3167 | Empty page table returns early. | +4/-3 |
| bd26bc8e | #3168 | Reject paged key count ≥ 2^31. | +9 |
| 328dd7b5 | #3173 | Clamp huge local-window sides (int32 overflow). | +31/-2 |
| f63c472c | #3150 | Clamp aux_tensors indices (score_mod/mask_mod only). | +37/-30 |
| c36859d9 | #3218 | Ragged TMA store base fix. Covers SM90 forward/backward and the MLA 2CTA forward, not the SM100 MHA forward. | +38/-15 |
| 307aa35d | #3085 | JIT cache dir fallback. | +32/-7 |
| 485f6e1d | #3205 | Move off deprecated CuTe DSL 4.7-4.9 names (25 files). | +158/-116 |

**Backward-only (training; no inference effect):** 2676a7be, db085f4e, b038b497, dbabcc78, dceb3741, c42f45c2 (+179/-7821; deletes the dedicated hd256 backward), c1709db7, 8fde8bf7, bd964619.

### Feature matrix for SM100/SM103 inference

"pin" means present at 1b32c5ca. "upstream" means present at 2676a7be.

| feature | pin | upstream | delta since sync |
|---|---|---|---|
| decode split-KV / `num_splits` (incl. per-sequence split metadata, `flash_fwd_combine.py`) | yes | yes | none for SM100. The SwapAB policy (`config.py` `SwapABPolicy`) is unchanged; `page_multiple=128` in both. |
| SwapAB decode kernel (`flash_fwd_decode_sm100.py`, SM100/SM103, hd 64/96/128 and 256/256) | yes | yes | f63c472c index clamp only |
| paged KV, TMA when page_size == tile_n, else cp.async (`paged_kv_non_tma`) | yes | yes | guards only (8fa1e462, bd26bc8e) |
| FP8 (tensorwise E4M3 Q/K/V + descales) on SM100 | yes (`interface.py:901`) | yes, but the API is now `TensorwiseScales` (9985a28e) | API break, not a feature |
| MXFP8 block-scaled attention | **no** | yes (69384e9c, 69832334, cc7b69d4; ~1.3k lines) | new; dense hd64/128 only |
| SM103 tunings (ex2-emu off on sm_103, register tables, SwapAB sm103 hd128 CTA target, `autotune.py` `103: sm_103a`) | yes | yes | only #3017's per-arch table (hd64/96 entries) |
| head_dim 256 forward (unified SM100 kernel, 1CTA/2CTA, split-KV) | yes | yes | c055bdc6 PackGQA selection fix |
| pack_gqa | yes | yes | c055bdc6 (hd256 odd ratio) |
| varlen | yes | yes | dynamic varlen route (8e3cb052, f4cc3b0b) |
| learnable sinks on SM100 | yes | yes | none on SM100 |
| softcap | yes | yes | none |
| CLC / persistent schedulers | yes (CLC, static persistent, `DynamicPersistentVarlenScheduler` class) | yes, plus **auto-routing** (`routes_dynamic_varlen`, `FwdConfig.use_dynamic_varlen`) | +264 lines |
| 2-CTA | yes | yes | none for forward MHA |
| block sparsity | yes | yes | none |

## Q3. Effect on our models

Model facts come from `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/.../config.json`:
- 40 layers, `full_attention_interval=4` (10 attention layers), `num_attention_heads=16`, `num_key_value_heads=2`, `head_dim=256`.
- GDN linear-attention layers: 32 value heads × 128.

Qwen3.8-Flash-Next is not cached locally. I took 24 q / 2 kv heads from the task brief; this is unverified.

### KV page size

- These are hybrid models, so vLLM sizes the attention block to cover the mamba page (`vllm/platforms/interface.py` `_align_hybrid_block_size`).
- Our logs show **block_size = 1088** (with FlashInfer).
- With FLASH_ATTN, the kernel alignment is `MultipleOf(16)` (`flash_attn.py:462`). I estimate the block size would be about 1056 without prefix caching, or 1088 with it. I computed this by hand, so it is uncertain.
- Either way it is **not a multiple of 128**, and that has three consequences in the pinned FA4:
  1. **SwapAB decode is rejected.** `SwapABPolicy.rejection`: "page size must be a multiple of 128".
  2. **Paged KV uses cp.async**, not TMA (`config.py:402` `paged_kv_non_tma = page_size not in (None, tile_n)`).
  3. **2CTA is not eligible** (`is_2cta_eligible` requires `page_size in (None, 128)`).

  Upstream main behaves the same on all three points.

### Is head_dim 256 supported on SM100/SM103 forward in the pin?

**Yes.**
- `_validate_head_dims` accepts `(256, 256)` (`interface.py:140-152`).
- The unified forward handles it: `flash_fwd_sm100.py` asserts "SplitKV is not supported for hdim_v >= 192 except 256".
- q_stage is 1 at hd256 (`config.py`, around line 1362).
- vLLM keeps FA4 for hd256 (`fa_utils.py:173-187`).
- `_is_fa4_supported` accepts any 10.x device (`flash_attn_interface.py:70-85`), so SM103 is included.

### Does decode (seqlen_q=1 per request, paged KV) use FA4 in the pin?

**Yes, FA4, but the standard SM100 kernel, not SwapAB.**
- vLLM always calls `flash_attn_varlen_func` with `cu_seqlens_q`, `seqused_k` and `block_table`, and `fa_version=4` on SM10x when FLASH_ATTN is selected (`vllm/vllm_flash_attn/flash_attn_interface.py:448-509`).
- With page size 1056/1088, SwapAB is rejected and decode runs on the standard kernel:
  - 128-row M tile, PackGQA, so only 8 (or 12) of the 128 rows are real.
  - cp.async paged K/V.
  - Split-KV. Under full CUDA graphs vLLM passes `num_splits=32` (`flash_attn_max_num_splits_for_cuda_graph`, `vllm/config/attention.py:67`). FA4 takes that as the split count, with per-sequence split metadata for varlen Q (`config.py` `needs_split_metadata`). Otherwise it passes 0, which means FA4's heuristic.
- **There is no fallback to FA2.**
- Upstream would behave identically for decode.
- If the KV block size were a multiple of 128 (for example 1152), SwapAB would be selected. The pin's own B300 sweep note says hd256 16/2 and 24/4 SwapAB ran 0.89-0.93x the time of the standard kernel (`config.py`, `SwapABPolicy.short_kv_*` comment).
- That would be a vLLM-side change: `get_supported_kernel_block_sizes` → `MultipleOf(128)` for hd256 FA4. It is independent of any sync. The gain is unmeasured by me.

### Per missing feature, does it affect our attention?

- **c055bdc6 (#3148): YES for the 24-q / 2-kv model (ratio 12) in prefill.**
  - The pin's `default_pack_gqa` (`config.py:603-622`) turns PackGQA off for hd256 when `128 % 12 != 0` and `max_seqlen_q > 128`, expecting 2CTA.
  - Our page size makes 2CTA impossible, so the call runs 1CTA unpacked, re-reading each KV head's K/V 12 times.
  - Upstream measured 15-28% slower than packed on B200 for 24/4 heads, 896-token pages, 200-650-token rows. That is exactly a Qwen-Next-like case.
  - No effect for the 16-q / 2-kv model (ratio 8: `128 % 8 == 0`, so it already packs). No effect on decode (`max_seqlen_q == 1`).
  - Easy cherry-pick: 1 file, +7/-1.
- **8e3cb052 + f4cc3b0b (#3087/#3086): probably YES for mixed prefill+decode steps.** I did not measure this.
  - It routes varlen calls with `num_splits == 1`, no caller scheduler metadata, and `batch * m_tile > total_q * rows`. One example is a ~300-token chunk plus ~31 decode slots.
  - vLLM passes no `scheduler_metadata` for FA4 (`aot_schedule = fa_version == 3`, `flash_attn.py:809`). Outside CUDA graphs it passes `num_splits=0`, so the route can apply.
  - Pure decode is excluded (one M tile per request). Pure large prefill is excluded (no excess slots).
  - A routed hd256 call is re-selected with PackGQA. That would also cover the ratio-12 case in padded batches.
- **bee0f085 (#3017): uncertain, likely small.** hd256 BF16 changes from cross-stage to same-stage stats release (q_stage=1). Output is bitwise identical, and no hd256 numbers were published.
- **MXFP8 (#3018/#3019/#3020/#3022): NO.**
  - Our KV is BF16.
  - MXFP8 is dense/unsplit and hd64/128 only; varlen, SwapAB and hd256 are rejected.
  - vLLM has no caller for it.
  - Tensorwise FP8 KV, if ever enabled, already exists in the pin.
- **9985a28e (#3027): NO perf effect.** It is a breaking API change: vLLM's `flash_attn_interface.py:503-505` passes `q/k/v_descale` and would need updating on the next sync.
- **SM107/SM120/SM80, MLA, backward, robustness fixes: NO** for B300 inference of these models.
  - 328dd7b5 matters only with huge local windows, which these models do not have.
  - 8fa1e462 and bd26bc8e are edge-case guards.
  - c36859d9 does not touch the SM100 MHA forward.
- **Already present in the pin, so nothing is missing:** decode split-KV, paged KV, hd256 forward, SM103 tunings, varlen, sinks, softcap, CLC/persistent, 2CTA, block sparsity.

## Uncertainties

- Qwen3.8-Flash-Next's head counts (24/2) come from the brief. They are not in a local config.
- The FLASH_ATTN block size (1056 vs. 1088) is computed by hand. The 1088 figure is from FlashInfer runs.
- No GPU measurements were taken. Impact sizes are upstream's figures (mostly B200), not B300 numbers for our shapes.
- All of this is moot under the default FLASHINFER backend on SM10x.
