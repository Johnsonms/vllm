# W8 — model-level A/B for candidate V (BV = 4 for an fp32 GDN state on B200)

**Result: the forward step gets faster by 0.4 % at 32-64 requests, 0.9 % at 128, 1.5 % at 256 and 2.0 % at 512, well
outside the noise (≤ 0.2 %). With MTP3 on there is no change, as expected. In serving, the p50 TPOT at c256 moves by
only 0.1 %, which is inside the noise, so by the decision rule this stops here. Mean TPOT and output throughput at c256
do improve by about 1 % in both rounds.**

Worker node, TEP4 on GPUs 4-7, vLLM 8399da3780, dummy weights, fp8 KV cache, fp8 indexer.
GPU 0 was in use by another workload throughout, and its load was **not constant**: before the forward runs it held
144-181 GB at 29-99 % utilisation; before the four serving runs it held 48, 54, 54 and 81 GB at 7, 33, 33 and 98 %.
GPUs 1-3 were idle and GPUs 4-7 were idle before every run. GPU 0 is on the other NUMA node (GPUs 0-3 vs 4-7), but
I cannot rule out an effect on the serving numbers. Local edits are reverted; the vLLM tree is clean.

## The two arms

Both arms ran from the same tree (`~/vllm-kopt`, launched with that directory as cwd). The edit is gated by an
environment variable read once at import, so `base` executes the original branch unchanged. Every process printed the
module path and its arm at import; all 12 printed the expected tree and arm. A direct check confirmed the launcher
picks BV = 8 in `base` and BV = 4 in `v` for B·HV > 64, and BV = 4 in both for B·HV ≤ 64.

Edit in `fused_recurrent_gated_delta_rule_packed_decode`, inside the SM100-family branch:

```python
elif initial_state.dtype == torch.float32:
    BV = 8
    if _KOPT_BV4 and current_platform.is_device_capability(100, device_id=dev.index):
        BV = 4          # arm v: fp32 state on capability 10.0 exactly (B200); GB300 keeps 8
```

## 1. Forward (`vllm bench forward`, `--iters 30 --samples 7 --routing balanced --kv-len-jitter 0`)

Order: base, v, base, v. Each arm-run is two processes: the decode cases without MTP, then the verify cases with
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Median step time in microseconds.

| case | base r1 | base r2 | v r1 | v r2 | mean Δ | spread within base | spread within v |
|---|---|---|---|---|---|---|---|
| d32 = 32*1@4096 | 8,541.9 | 8,552.0 | 8,515.3 | 8,513.6 | **-0.38 %** | 0.12 % | 0.02 % |
| d64 = 64*1@4096 | 10,889.8 | 10,867.3 | 10,832.3 | 10,832.4 | **-0.42 %** | 0.21 % | 0.00 % |
| d128 = 128*1@4096 | 13,681.3 | 13,669.0 | 13,545.2 | 13,547.0 | **-0.94 %** | 0.09 % | 0.01 % |
| d256 = 256*1@4096 | 17,020.4 | 17,022.6 | 16,764.4 | 16,766.5 | **-1.50 %** | 0.01 % | 0.01 % |
| d512 = 512*1@2048 | 23,648.8 | 23,659.1 | 23,192.1 | 23,194.1 | **-1.95 %** | 0.04 % | 0.01 % |
| mtp_c16 = 16*4@4096:3 (MTP3) | 11,984.8 | 11,990.6 | 11,992.8 | 11,990.5 | +0.03 % | 0.05 % | 0.02 % |
| mtp_c64 = 64*4@4096:3 (MTP3) | 17,688.3 | 17,702.5 | 17,700.5 | 17,706.7 | +0.05 % | 0.08 % | 0.04 % |

- **Deviation from the task:** d512 uses a 2,048-token context instead of 4,096. The manager's earlier tip said 512
  requests at 4,096 may not fit the KV cache, and a failing case hangs the run (W4), so I used the safe length.
- The gain grows with the request count, as the kernel's share of the step does. At 256 requests it is 1.5 %, a little
  above my estimate of 1.1 % from the kernel share.
- **MTP3: no change.** The verify step runs the C++ `gdn_decode_post_conv_mtp_kernel`, not this Triton kernel
  (W7a caveat). The drafter does not go through the changed branch either, or the MTP cases would have moved.

## 2. Serving (`vllm serve … --load-format dummy`, TEP4, fp8 KV, fp8 indexer, no MTP)

Flags: `-tp 4 --enable-expert-parallel --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8
--language-model-only --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384
--gpu-memory-utilization 0.85 --no-enable-prefix-caching`. Load: `vllm bench serve`, random dataset, 512 in / 512 out,
`--ignore-eos`, 3 x concurrency prompts, 16 warm-ups. Order: base, v, base, v; one server start per run.
**MTP was off**, because with MTP on the changed kernel does not run.

| concurrency | metric | base r1 | base r2 | v r1 | v r2 | mean Δ | spread within base | spread within v |
|---|---|---|---|---|---|---|---|---|
| 64 | TPOT p50 (ms) | 11.145 | 11.089 | 11.104 | 11.059 | -0.32 % | 0.50 % | 0.41 % |
| 64 | TPOT mean (ms) | 11.170 | 11.116 | 11.115 | 11.093 | -0.35 % | 0.48 % | 0.20 % |
| 64 | output tokens/s | 5,447 | 5,501 | 5,451 | 5,487 | -0.05 % | 0.99 % | 0.64 % |
| 256 | **TPOT p50 (ms)** | 18.095 | 18.069 | 18.150 | 17.972 | **-0.12 %** | 0.14 % | 0.99 % |
| 256 | TPOT mean (ms) | 18.123 | 18.044 | 17.957 | 17.862 | -0.96 % | 0.44 % | 0.53 % |
| 256 | output tokens/s | 13,222 | 13,220 | 13,325 | 13,384 | +1.01 % | 0.02 % | 0.44 % |

All requests completed in every run (192 at c64, 768 at c256), none failed.

- **c256, TPOT p50: -0.12 %, inside the noise** (the two `v` runs differ by 1 %). By the decision rule this is
  "< 0.5 % or inside the noise", so I stop here.
- **c256, mean TPOT and output throughput: about 1 % better in both rounds**, and each `v` run beats both `base` runs
  on these two metrics. That agrees with the forward result (-1.5 % per step at 256 requests), but two runs per arm is
  a thin basis.
- **c64: no measurable change.** All differences are below the run-to-run spread.

## Reading

- The change is real at step level and bit-identical (W7a), but small: about 1.5 % of a 256-request step, nothing under
  MTP3, and at most about 1 % in serving at c256.
- If the target deployment runs MTP3, candidate V gives nothing there.
- If more serving evidence is wanted, the next step is more alternating rounds at c256 (the p50 needs more than two
  runs per arm to resolve 1 %) and a c512 point, where the forward gain is 2 %.
