import json, sys
silu = {(r["I"], r["M"]): r for r in map(json.loads, open("/scratch/kopt-w21/silu_supp2.jsonl"))}
for model in ("Qwen3-8B-FP8", "Qwen3-32B-FP8"):
    rows = [json.loads(l) for l in open(f"/scratch/kopt-w21/bench_{model}.jsonl")]
    print(f"\n### {model}: plain GEMM, prequantized inputs -> bf16 (us; speedup = DeepGEMM / QuACK)\n")
    print("| op (N x K) | M | DeepGEMM | QuACK MXFP8 | speedup | err vLLM | err QuACK | QuACK quant-out | quant-out err |\n|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        qo = r.get("q_gemm_qout_err"); flag = " **WRONG**" if qo is None or not (qo < 0.06) else ""
        print(f"| {r['op']} ({r['N']}x{r['K']}) | {r['M']} | {r['v_gemm_us']} | {r['q_gemm_us']} | {r['v_gemm_us']/r['q_gemm_us']:.2f} | {r['v_gemm_err']} | {r['q_gemm_err']} | {r.get('q_gemm_qout_us')} | {qo}{flag} |")
    print(f"\n### {model}: input quantization of the bf16 activations (us)\n\n| op | M | vLLM packed 1x128 kernel | QuACK `quantize` (compiled torch, 1x32) |\n|---|---|---|---|")
    for r in rows: print(f"| {r['op']} (K={r['K']}) | {r['M']} | {r['v_quant_in_us']} | {r.get('q_quant_in_us')} |")
    print(f"\n### {model}: gate_up chain (us). vLLM = DeepGEMM + silu-mul (Inductor kernel, as in serving) [+ 1x128 quant of h]; QuACK = gated GEMM with SwiGLU epilogue [+ quant]\n")
    print("| M | DeepGEMM | silu Inductor | vLLM GEMM+silu | QuACK gated | speedup | err vLLM | err QuACK | vLLM +quant | QuACK gated + vLLM quant kernel | speedup | QuACK gated + QuACK quantize |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["op"] != "gate_up": continue
        s = silu[(r["N"] // 2, r["M"])]; adj = s["inductor_us"] - s["custom_op_us"]
        a = r["v_gemm_silu_us"] + adj; aq = r["v_gemm_silu_quant_us"] + adj
        print(f"| {r['M']} | {r['v_gemm_us']} | {s['inductor_us']} | {a:.2f} | {r.get('q_gated_us')} | {a/r['q_gated_us']:.2f} | {r['v_gemm_silu_err']} | {r.get('q_gated_err')} | {aq:.2f} | {r.get('q_gated_vquant_us')} | {aq/r['q_gated_vquant_us']:.2f} | {r.get('q_gated_quant_us')} |")
    fails = {k: v for r in rows for k, v in r.items() if k.endswith("_fail")}
    if fails: print("\nfailures:", json.dumps(fails)[:1500])
