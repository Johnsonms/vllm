# W6 — prefill non-determinism above 2,048 tokens

**Not a wrong-data bug, as far as tested.** The source is the sparse-attention (QSA) indexer's top-k: the FlashInfer
`gvr_2` backend returns a correct top-k, but breaks ties at the cut-off differently from call to call and returns the
indices in a different order each time. Root cause is confirmed to that level; why the kernel is not stable was not
investigated.

Worker node, GPUs 4-7 (idle before each run), vLLM 8399da3780, TEP4, dummy weights, one prefill request per case.
Local debug edits only; all reverted, vLLM tree clean.

## 1. Threshold

| prompt tokens | eager repeat (rel. RMS) | max abs | graph vs eager | graph deterministic |
|---|---|---|---|---|
| 2,048 | 0 | 0 | 0 | yes |
| 2,049 | 0 | 0 | 0 | yes |
| 2,056 | 0.0038 | 0.277 | 0.0034 | **no** |
| 2,304 | 0.0181 | 0.445 | 0.0167 | **no** |
| 4,096 | 0.0163 | 0.355 | 0.0185 | **no** |

This matches the model config: `indexer_budget: 2048` and `indexer_compress_ratio: 4` give a block top-k of 512.
The top-k only has something to drop once a token sees more than 512 compressed blocks. At 2,049 tokens no row is
selective yet; at 2,056 tokens 5 rows per call are, and the output starts to differ.
Not tested: dependence on `--max-num-batched-tokens` or chunked prefill.

## 2. First divergent module

Method: forward hooks on every module, recording two checksums of each output for the two eager runs from the same
restored state, compared in call order (811 module calls per run).

- 2,048 and 2,049 tokens: 0 of 811 outputs differ, on all 4 ranks.
- 4,096 tokens: 663 of 811 differ, on all 4 ranks. The **first** is
  `language_model.model.layers.3.self_attn.indexer` (`QSAIndexer`, output shape `(4096, 2052)`, the selected indices).
  Everything before it is identical, including layers 0-2. Next in order are that layer's `o_proj`, the attention
  output, then the hyper-connection and MoE of layer 3, and so on downstream.

So the GDN layers, MoE, hyper-connections and the all-reduce are not the origin.

## 3. What the top-k does

Method: inside `qsa_indexer._topk`, for prefill calls in eager mode, compare the `gvr_2` result with a float64
`torch.topk` reference on the same logits, and call `gvr_2` a second time on an identical copy of the logits.
Totals over all 4 ranks:

| logits shape (tokens x blocks) | calls | selective rows | selected below the cut-off | missed above the cut-off | duplicates / invalid | rows with ties at the cut-off | set differs on 2nd call | order differs on 2nd call |
|---|---|---|---|---|---|---|---|---|
| 2,049 x 576 | 144 | 0 | — | — | — | — | — | — |
| 2,056 x 576 | 144 | 720 | 0 | 0 | 0 | 720 (up to 50 tied values) | 347 | 716 |
| 4,096 x 1,024 | 144 | 294,480 | 0 | 0 | 0 | 18,664 (up to 63) | 10,813 | 294,391 |
| 8,191 x 2,048 (warm-up) | 48 | 294,720 | 0 | 0 | 0 | 6,704 (up to 67) | 3,981 | 294,699 |

- **The selection is always a valid top-k:** nothing below the k-th value is picked, nothing above it is missed, no
  duplicates, no out-of-range indices, in both calls.
- **The set changes only through ties:** rows whose set differs (10,813 at 4,096 tokens) are fewer than rows that have
  ties at the cut-off (18,664).
- **The order changes in practically every selective row**, with identical input.
- The kernel does not modify its input logits.

## 4. Verdict

**Expected-type non-determinism (tie-break and output order), not wrong data, buffer overflow or a race on values.**
The effect is large in this benchmark for two reasons I could confirm only in part:

- Ties are plentiful here: up to 63 exactly equal scores at the cut-off in one row. Dummy weights very likely inflate
  this. **Tie frequency with real weights was not measured**, so the size of the effect in production is unknown.
- Whether the changed order alone changes the result (through summation order in the sparse attention) or only the
  changed sets do was not separated.

## 5. One observation that did not reproduce

In the top-k check run, 4 of the 144 calls at 4,096 tokens had NaNs in the logits (about 310,000 values each). A
follow-up run that recorded NaN positions found none in 288 calls at 4,096 tokens. My reading: the cells beyond each
row's visible blocks are left unwritten, and they showed leftovers of my own debug tensors (a float64 `-inf` reads as a
NaN in fp32). That would be harmless, since the top-k respects the visible count, but I did not record visible versus
padding in the run where the NaNs appeared, so this is an inference.

## Minimal repro

Unmodified vLLM 8399da3780, 4 GPUs:

```bash
torchrun --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 --master-port=29701 --nproc-per-node=4 \
  -m vllm.entrypoints.cli.main bench forward --model Qwen/Qwen3.8-Flash-Next-FP8 -tp 4 --enable-expert-parallel \
  --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --max-num-seqs 256 --max-model-len 16384 \
  --gpu-memory-utilization 0.7 --language-model-only --kv-len-jitter 0 --routing balanced --iters 3 --samples 2 \
  --case t2049=2049 --case t2056=2056 --output-json out.json
```

In `out.json`, `eager_repeat.relative_rms` is 0 for `t2049` and about 0.004 for `t2056`;
`graph_deterministic` is true and false.

## Suggested next steps (not done)

- Rerun 4,096 tokens with `sparse_indexer_topk_backend` set to `cooperative` or `persistent`: if the result becomes
  exact, the instability is specific to `gvr_2`.
- Measure tie counts with real weights. If ties are rare there, this is a benchmark artifact of dummy weights.
- If reproducibility matters for A/B quality runs, a stable tie-break (lowest index wins) in the top-k would fix it.
- Not tested at TP1, so whether tensor parallelism plays any part is unknown; the first divergent module suggests not.
