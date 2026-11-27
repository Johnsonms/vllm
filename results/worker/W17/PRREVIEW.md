# W17 PRREVIEW — shared-expert gate skinny plan ("W15" PR), for manager review (2026-11-27 19:50Z)

- **Commit:** `383e366bcb1d4bf55e6c63d24c8aeacd6abfed96`, tree `3e82101d8d898145da2e32691d41e7dc563b4b31`, one commit on main `361f362033`. Local only; nothing pushed.
- **Identity:** author and committer = owner identity (checked with the `%an <%ae> | %cn <%ce>` log line), `Assisted-by: Claude Code` directly above `Signed-off-by`, no Co-authored-by.
- **Diffstat:**

     .buildkite/intel_jobs/basic_correctness_intel.yaml |   2 +-
     .buildkite/intel_jobs/test-intel.yaml              |   6 +-
     .../configs/Meta-Llama-3-8B-QQQ.yaml               |   4 +-
     .../configs/Minitron-4B-Base-FP8.yaml              |   4 +-
     .../scripts/hardware_ci/run-cpu-test-ppc64le.sh    |   4 +-
     .../scripts/hardware_ci/run-intel-ci-test.sh       |   6 +-
     .buildkite/test_areas/distributed.yaml             |   4 +-
     docs/contributing/model/tests.md                   |   2 +-
     docs/models/structured_decisions_models.md         |  10 +-
     tests/AGENTS.md                                    |   2 +-
     .../cpu_offload/test_cpu_offload.py                |   8 +-
     tests/basic_correctness/memory/cumem/test_cumem.py |  46 ++----
     .../memory/sleep_mode/test_sleep_mode.py           |  32 ++--
     .../models/test_basic_correctness.py               |   4 +-
     tests/compile/fullgraph/test_full_graph.py         |   2 +-
     tests/config/base_model_arch_groundtruth.json      |   6 +-
     tests/config/draft_model_arch_groundtruth.json     |   8 +-
     tests/config/test_model_arch_config.py             |  12 +-
     .../test_speculative_draft_limits_max_model_len.py |   6 +-
     ...st_speculative_draft_max_position_embeddings.py |   8 +-
     ...test_stop_string_while_stop_model_terminates.py |   4 +-
     tests/distributed/test_eplb_spec_decode.py         |   2 +-
     tests/distributed/test_pipeline_parallel.py        |   8 +-
     tests/distributed/test_pp_cudagraph.py             |   2 +-
     tests/distributed/test_sharded_rdt_plan.py         |   2 +-
     tests/engine/test_arg_utils.py                     |   2 +-
     tests/entrypoints/cohere/test_api_router.py        |   2 +-
     .../entrypoints/cohere/test_cohere_chat_message.py |   2 -
     .../entrypoints/cohere/test_serving_conversion.py  |  35 +----
     tests/entrypoints/conftest.py                      |   6 +-
     .../structured_decisions/test_decisions.py         |   2 +-
     .../structured_decisions/test_slot_decision_e2e.py |  11 +-
     .../structured_decisions/test_slot_strategy.py     |  13 --
     .../structured_decisions/test_strategies.py        |  49 ++-----
     .../launchers/api_server/test_multi_api_servers.py |   2 +-
     tests/entrypoints/launchers/test_dp_supervisor.py  |  26 ++--
     tests/entrypoints/launchers/test_run_batch.py      |  16 +-
     tests/entrypoints/launchers/test_shutdown.py       |   2 +-
     tests/entrypoints/llm/test_collective_rpc.py       |   4 +-
     .../entrypoints/llm/test_struct_output_generate.py |   2 +-
     .../openai/chat_completion/test_chat.py            |   2 +-
     .../openai/decisions/test_decisions_e2e.py         |  36 -----
     tests/entrypoints/pooling/basic/test_encode.py     |   2 +-
     .../pooling/basic/test_tiling_engine.py            |   2 +-
     tests/entrypoints/pooling/classify/test_offline.py |   2 +-
     tests/entrypoints/pooling/classify/test_online.py  |   4 +-
     .../pooling/classify/test_online_vision.py         |   2 +-
     .../pooling/embed/test_cohere_online.py            |   2 +-
     .../pooling/embed/test_correctness_mteb.py         |   2 +-
     tests/entrypoints/pooling/embed/test_offline.py    |   2 +-
     tests/entrypoints/pooling/embed/test_online.py     |   2 +-
     .../pooling/embed/test_online_dimensions.py        |   2 +-
     .../pooling/embed/test_online_long_text.py         |   4 +-
     .../pooling/scoring/test_bi_encoder_offline.py     |   2 +-
     .../pooling/scoring/test_cross_encoder_offline.py  |   2 +-
     .../pooling/token_classify/test_offline.py         |   2 +-
     .../pooling/token_classify/test_online.py          |   2 +-
     .../pooling/token_embed/test_offline.py            |   2 +-
     .../entrypoints/pooling/token_embed/test_online.py |   2 +-
     .../entrypoints/rl/weight_transfer/test_offline.py |   2 +-
     .../scale_out/derender/test_derender.py            |   2 +-
     .../scale_out/derender/test_derender_stream.py     |   8 +-
     .../derender/test_generate_logprobs_conversion.py  |   3 +-
     .../derender/test_logprobs_endpoint_parity.py      |   3 +-
     .../scale_out/render/test_launch_render.py         |   2 +-
     tests/entrypoints/scale_out/render/test_render.py  | 162 ++++++++-------------
     .../token_in_token_out/test_generate_stream.py     |  65 ++++-----
     .../token_in_token_out/test_logprobs_render.py     |   2 -
     .../scale_out/token_in_token_out/test_mm_serde.py  |   6 +-
     .../scale_out/token_in_token_out/test_protocol.py  |  18 +--
     .../test_serving_multimodal_tokens.py              |  17 +--
     .../token_in_token_out/test_serving_tokens.py      |  10 +-
     .../token_in_token_out/test_tokens_logprobs.py     |  11 --
     .../serve/instrumentator/test_metrics.py           |   4 +-
     .../entrypoints/serve/lora/test_serving_models.py  |   2 +-
     .../serve/middleware/test_optional_middleware.py   |   2 +-
     tests/entrypoints/serve/sagemaker/conftest.py      |   2 +-
     .../translation/test_translation_validation.py     |   2 +-
     .../tool_parsers/test_hermes_tool_parser.py        |   2 +-
     .../configs/Kimi-K3-pruned75-DSpark-AITER-TP4.yaml |   2 +-
     .../gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml |   2 +-
     .../humming/Qwen3-0.6B-MXFP8-humming-act-fp8.yaml  |   2 +-
     .../configs/humming/Qwen3-0.6B-MXFP8-humming.yaml  |   2 +-
     ...wen3-30B-A3B-MXFP8-CT-fi-cutedsl-deepep-ll.yaml |   2 +-
     tests/jit_monitor/test_no_runtime_jit.py           |   8 +-
     tests/kernels/moe/test_ocp_mx_moe.py               |  12 +-
     tests/kernels/test_bf16_skinny_gemm.py             |   8 +
     tests/lora/conftest.py                             |  80 ++++++----
     tests/lora/test_lora_functions.py                  |   2 +-
     tests/lora/test_quant_model.py                     |   8 +-
     tests/lora/test_qwen3_with_multi_loras.py          |   6 +-
     tests/lora/test_transformers_model.py              |   2 +-
     tests/model_executor/model_loader/test_reload.py   |   4 +-
     .../model_executor/test_model_load_with_params.py  |   4 +-
     .../test_sleep_mode_tensor_ownership.py            |   2 +-
     .../amd_specialized_decode/test_engine_fault.py    |   4 +-
     tests/models/kimi_k3/test_prefix_cache.py          |   4 +-
     tests/models/language/generation/test_common.py    |   2 +-
     tests/models/language/generation/test_hybrid.py    |   8 +-
     .../pooling/test_auto_prefix_cache_support.py      |   6 +-
     .../models/language/pooling/test_classification.py |   2 +-
     tests/models/language/pooling/test_embedding.py    |  12 +-
     tests/models/language/pooling/test_head_dtype.py   |   2 +-
     .../language/pooling/test_max_tokens_per_doc.py    |   2 +-
     .../test_multilabel_classification_support.py      |   2 +-
     .../pooling/test_pooler_config_init_behaviour.py   |   8 +-
     .../language/pooling/test_token_classification.py  |  10 +-
     .../pooling_mteb_test/test_cross_encoder.py        |   2 +-
     .../models/language/pooling_mteb_test/test_gte.py  |  22 ++-
     .../language/pooling_mteb_test/test_intfloat.py    |  16 +-
     tests/models/multimodal/generation/test_common.py  |   2 +-
     tests/models/multimodal/pooling/test_colqwen3_5.py |   4 +-
     .../multimodal/pooling/test_colultravbert.py       |   2 +-
     .../models/multimodal/pooling/test_dse_qwen2_vl.py |   2 +-
     tests/models/multimodal/pooling/test_llava_next.py |   2 +-
     .../processing/test_transformers_video.py          |   4 +-
     tests/models/quantization/test_awq.py              |   2 +-
     tests/models/quantization/test_gptq_marlin.py      |   4 +-
     tests/models/registry.py                           |  90 +++++++-----
     tests/models/transformers/test_backend.py          |   8 +-
     tests/platforms/test_cpu.py                        |   2 +-
     .../bitsandbytes/test_bitsandbytes.py              |   6 +-
     .../lora_resolvers/test_filesystem_resolver.py     |   4 +-
     .../lora_resolvers/test_hf_hub_resolver.py         |   6 +-
     tests/quantization/test_auto_round.py              |   8 +-
     tests/quantization/test_compressed_tensors.py      |   4 +-
     tests/quantization/test_configs.py                 |  16 +-
     tests/quantization/test_cpu_wna16.py               |   2 +-
     tests/quantization/test_modelopt.py                |   4 +-
     tests/quantization/test_online.py                  |   8 +-
     tests/quantization/test_quark.py                   |   6 +-
     tests/rocm/test_moe_padding.py                     |   8 +-
     tests/rocm/test_moe_weight_replay.py               |  10 +-
     tests/samplers/test_beam_search.py                 |   2 +-
     tests/samplers/test_no_bad_words.py                |   2 +-
     tests/standalone_tests/python_only_compile.sh      |   2 +-
     tests/test_config.py                               |  34 ++---
     tests/test_pooling_params.py                       |   2 +-
     tests/tool_use/utils.py                            |   2 +-
     .../test_indexer_deepseek_v4_slot_mapping.py       |   2 +-
     .../test_rocm_paged_mxfp4_indexer_plan.py          |   2 +-
     tests/v1/core/test_scheduler_e2e.py                |   2 +-
     tests/v1/distributed/test_async_llm_dp.py          |   4 +-
     tests/v1/distributed/test_dense_dp_world_size.py   |   2 +-
     tests/v1/distributed/test_eagle_dp.py              |   2 +-
     tests/v1/e2e/general/test_context_length.py        |   4 +-
     .../acceptance_rates/dflash/test_dflash.py         |   2 +-
     tests/v1/e2e/spec_decode/draft_model/test_async.py |   4 +-
     tests/v1/e2e/spec_decode/draft_model/test_lora.py  |   4 +-
     .../spec_decode/eagle/test_eagle_correctness.py    |   8 +-
     tests/v1/e2e/spec_decode/mtp/deepseek/test_mtp.py  |   2 +-
     tests/v1/e2e/spec_decode/test_mtp_parallel_load.py |   4 +-
     tests/v1/engine/test_engine_core.py                |   2 +-
     tests/v1/engine/test_preprocess_error_handling.py  |   2 +-
     .../kv_connector/unit/test_offloading_connector.py |   4 +-
     tests/v1/sample/test_masked_token_ids.py           |   2 +-
     tests/v1/sample/test_sampling_params_e2e.py        |   2 +-
     tests/v1/shutdown/test_delete.py                   |   2 +-
     tests/v1/shutdown/test_forward_error.py            |   2 +-
     tests/v1/shutdown/test_startup_error.py            |   6 +-
     tests/v1/spec_decode/test_max_len.py               |   2 +-
     tests/watermarking/golden_candidates.py            |   3 -
     tests/watermarking/test_compatibility.py           |   5 -
     tests/watermarking/test_engine_compatibility.py    |   4 +-
     tests/watermarking/test_parallel_compatibility.py  |   2 +-
     tests/watermarking/test_watermarking.py            |  58 ++------
     tools/pre_commit/mypy.py                           |  23 ---
     vllm/config/compilation.py                         |   4 +-
     vllm/entrypoints/openai/decisions/adapters.py      |  14 +-
     .../entrypoints/openai/decisions/question_types.py |  10 +-
     vllm/entrypoints/openai/decisions/strategies.py    |  16 +-
     vllm/models/qwen4_exp/nvidia/low_latency_gemm.py   |  15 ++
     172 files changed, 629 insertions(+), 856 deletions(-)

- **What the diff is, in words:** 11 lines plus a 2-line comment for key (1, 2560) at the end of the Flash-Next SM100 table (the entries in `entries.json`, written in the table's short form), and one 6-line test that asserts the key's M list and that each config's row count equals its M. The existing table-driven SM100 test picks up the 11 new cases by itself.

## Evidence

**Unit tests.** `pytest tests/kernels/test_bf16_skinny_gemm.py -q` on one B200: PR 711 passed / 471 skipped; base 699 passed / 471 skipped. The 12 extra are the 11 new table cases plus the new test; the new test cannot pass on the base (the key does not exist). `pre-commit run --files` on the 2 files: all passed, no file modified.

**Kernel microbench** (W15 / W16 method: CUDA graph, 4 seeds x 3 interleaved rounds vs cuBLAS, min-max over seeds):

| M | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speedup | 2.23-2.34 | 2.44-2.58 | 2.34-2.49 | 2.09-2.27 | 2.16-2.25 | 2.03-2.09 | 2.19-2.52 | 1.84-2.06 | 1.74-1.78 | 1.70-1.79 | 1.64-1.76 |

cuBLAS 4.1-5.0 us per call, skinny 1.9-2.9 us. Max relative error vs an fp32 reference equals cuBLAS on every entry. No entry is slower.

**End to end on this SHA** (`ab/`). Qwen3.8-Flash-Next-FP8, real weights, TEP4 on 4 B200, recipe flags, no speculative decoding, random 512 in / 512 out, no prefix cache, compile cache off, "Directly load AOT" 0 times in all 6 logs, loaded file path and HEAD logged per server. Two TEP4 servers do not fit on the free GPUs, so the A/B is interleaved restarts A B A B A B on the same 4 GPUs: 3 starts per arm; after each start a discarded warm-up, then 4 bursts of 5 x c prompts per concurrency. Per start = median over bursts of median TPOT; delta = mean of per-start medians.

| requests | base per start (ms) | base spread | PR per start (ms) | PR spread | delta | PR starts faster than every base start |
|---|---|---|---|---|---|---|
| 1 | 3.984 3.995 3.991 | 0.28 % | 3.977 3.978 3.993 | 0.40 % | -0.19 % | 2 of 3 |
| 2 | 4.484 4.493 4.466 | 0.59 % | 4.465 4.468 4.470 | 0.10 % | -0.29 % | 1 of 3 |
| 4 | 5.316 5.320 5.304 | 0.29 % | 5.241 5.236 5.247 | 0.21 % | **-1.35 %** | 3 of 3 |
| 8 | 6.080 6.091 6.078 | 0.22 % | 5.977 5.983 5.986 | 0.14 % | **-1.67 %** | 3 of 3 |
| 16 | 6.623 6.637 6.624 | 0.21 % | 6.571 6.562 6.587 | 0.38 % | **-0.82 %** | 3 of 3 |
| 32 | 8.038 8.046 8.022 | 0.30 % | 8.030 8.021 8.025 | 0.12 % | -0.13 % | 1 of 3 |

- **The gain is real at 4, 8 and 16 requests and not shown at 1, 2 and 32.**
- **It is about half of the W16 number.** W16 (main 0fbba2aa71, same 11 entries) had -0.7 / -2.3 / -3.0 / -0.9 % at 2 / 4 / 8 / 16. The base TPOT is the same within 0.8 % on both mains, so the candidate got slower relative to W16, not the base faster. I have not found the cause. Main moved 12 files of the Flash-Next model code between the two SHAs. The body below uses only the numbers of this SHA.

**Greedy, 32 single prompts x 256 tokens:** identical, 0 of 32 differ in all 9 base-vs-PR start pairs (base vs base 0, PR vs PR 0). This differs from W16 on the older main, where 2 of 32 differed. The kernel is still a different summation order than cuBLAS, so I do not claim bit-identical output in the body.

**Batched greedy:** not used. On TEP4 the 256-prompts-in-one-request output is not reproducible between two starts of the same tree (W16).

**GSM8K, 5-shot, 1319 questions, temperature 0, 64 concurrent, 8 runs per arm** (3 + 3 + 2 over the three starts; per-question dumps in `ab/*/gsm_perq_*.json`):
- Correct — base: 1273 1277 1281 1276 1274 1277 1277 1271 (mean 0.9672, sd 3.1 questions). PR: 1276 1275 1272 1280 1270 1276 1277 1278 (mean 0.9670, sd 3.2).
- Per-question flips between two runs: base vs base 7-19 (median 12); PR vs PR 7-18 (median 13); base vs PR 6-17 (median 12).
- No question differs by 4 or more of 8 between the arms (largest difference 3 of 8).

**Not measured:** TP8 / TEP8; B300 / SM103; speculative decoding on; prefill; any base that includes #69907. The compiled extension on this node is from an older wheel: main changed two CUDA sources since (the DSA top-k and one HC kernel file); both arms use the same binary.

## Duplicate check
Searched open and merged PRs for "skinny gemm", "Qwen3.8 skinny", "Flash-Next GEMM", "SM100 small GEMM", "low_latency_gemm". No PR adds (1, 2560) to the SM100 table. Related:
- #69736 (merged): the SM100 table this extends.
- #57848 (open): SM103 router lines in the same file; different table, different shape.
- #66865 (open): GB300 specialized decode; SM103 only.
- #69907 (open, no review): makes FlashInfer cute-dsl the default small-batch BF16 GEMM in the generic unquantized linear method. Today the gate is not in the SM100 table, so after #69907 the gate would run on that kernel in the base. This PR would then move it to the skinny kernel at M <= 16 and to torch linear at 17-32. **If #69907 merges first, the A/B must be repeated against it.**

## Title
`[Perf] SM100 skinny GEMM plan for the Qwen3.8-Flash-Next shared-expert gate`

## Body

### Overview
The shared-expert gate of Qwen3.8-Flash-Next is a BF16 linear with a single output row, local weight shape `(N, K) = (1, 2560)`. It runs 48 times per decode step on each rank. On SM100 this shape is not in the model's low-latency GEMM table, so it runs on the standard linear backend (cuBLAS), at 4-5 us per call.

This PR adds measured CuTe DSL skinny GEMM configurations for that shape at `M = 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16`. These are the decode CUDA graph sizes up to 16. It reuses the existing kernel and dispatch guards; only the table grows.

Not affected: batches above 16 rows, other shapes, other GPUs (the SM103, SM90 and SM121 tables are unchanged), batch-invariant mode.

### Results
End to end, Qwen3.8-Flash-Next-FP8, TEP4 on 4x B200, no speculative decoding, random 512 in / 512 out.
- Method: the two arms are restarted alternately on the same 4 GPUs (A B A B A B, 3 starts per arm). After each start: one discarded warm-up, then 4 bursts of 5 x c prompts per concurrency. The number per start is the median over bursts of the median TPOT.
- Compile cache off; main at `361f362033`.

| concurrent requests | main TPOT (ms), 3 starts | PR TPOT (ms), 3 starts | change |
|---|---|---|---|
| 1 | 3.984 3.995 3.991 | 3.977 3.978 3.993 | -0.2 % |
| 2 | 4.484 4.493 4.466 | 4.465 4.468 4.470 | -0.3 % |
| 4 | 5.316 5.320 5.304 | 5.241 5.236 5.247 | -1.4 % |
| 8 | 6.080 6.091 6.078 | 5.977 5.983 5.986 | -1.7 % |
| 16 | 6.623 6.637 6.624 | 6.571 6.562 6.587 | -0.8 % |
| 32 | 8.038 8.046 8.022 | 8.030 8.021 8.025 | -0.1 % |

At 4, 8 and 16 requests every PR start is faster than every main start. At 1, 2 and 32 the difference is inside the start-to-start spread.

Kernel level, skinny vs cuBLAS in a CUDA graph (4 seeds, 3 interleaved rounds): 1.64-2.58x over the 11 sizes; no size is slower.

Accuracy. The kernel sums in a different order than cuBLAS, so the output is not guaranteed bit-identical.
- Greedy, 32 single prompts x 256 tokens: identical to main.
- GSM8K (5-shot, 1319 questions, temperature 0), 8 runs per arm: 0.9672 on main, 0.9670 with the PR. Two runs of the same arm differ on 7-19 questions; a main run and a PR run differ on 6-17. No question differs by 4 or more of 8 runs between the arms.

Entries that were tried and left out:
- The router `(512, 2560)` at M = 16 wins at kernel level (1.2x) and loses end to end (+3.7 % TPOT at 16 requests).
- Extra M for the already-listed projections and the LM head showed no end-to-end effect.
- M = 7, 9, 11, 13 and 15 are not decode CUDA graph sizes.

Not measured: TP8, SM103, speculative decoding, prefill.

### Testing
- `pytest tests/kernels/test_bf16_skinny_gemm.py` on a B200: 711 passed, 471 skipped (main: 699 passed, 471 skipped). The existing table-driven SM100 test covers the 11 new entries; one new test pins the list of sizes.
- `pre-commit run --files` on the changed files passes.

### Related PRs
- #69736 added the SM100 table this extends.
- #57848 tunes the SM103 router lines in the same file; no overlap.
- #69907 would make FlashInfer cute-dsl the default small-batch BF16 GEMM for layers on the generic linear method. Layers in this table do not use that method, so with both PRs the gate uses the skinny kernel up to 16 rows. The two have not been compared on this shape.

### AI assistance
Claude Code was used to run the sweeps and benchmarks and to draft the change and this description. The submitter reviewed every changed line.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
