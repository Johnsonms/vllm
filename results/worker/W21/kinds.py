"""W21 step 1: kernel time by kind from a torch-profiler trace. usage: kinds.py TRACE.json.gz STEPS LAYERS"""
import gzip, json, re, sys, collections
f, steps, layers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ev = [e for e in json.load(gzip.open(f))["traceEvents"] if e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
def kind(n):
    for k, rx in [("DeepGEMM fp8 gemm", r"deep_gemm|sm100_fp8_gemm|fp8_gemm"), ("FP8 group quant", r"per_token_group_quant"),
                  ("silu_and_mul", r"act_and_mul|silu"), ("attention (FlashInfer)", r"fmha|flashinfer|Fmha|trtllm"),
                  ("rms norm", r"norm|Norm"), ("rope / cache write", r"rope|rotary|reshape_and_cache"),
                  ("cuBLAS (LM head)", r"nvjet|cutlass_80|gemv|splitK|sgemm|gemm"), ("sampling", r"ampl|SoftMax|softmax|topk|TopK|sort"),
                  ("torch elementwise/index/copy", r"at::native|elementwise|index|copy|Fill|Memcpy|Memset")]:
        if re.search(rx, n): return k
    return "other"
agg = collections.defaultdict(lambda: [0, 0]); names = collections.defaultdict(lambda: [0, 0]); gi = 0; per = collections.defaultdict(lambda: [0, 0])
span = ev[-1]["ts"] + ev[-1]["dur"] - ev[0]["ts"]
for e in ev:
    k = kind(e["name"]); agg[k][0] += e["dur"]; agg[k][1] += 1; names[(k, e["name"][:90])][0] += e["dur"]; names[(k, e["name"][:90])][1] += 1
    if k == "DeepGEMM fp8 gemm":
        m = ("qkv", "o", "gate_up", "down")[gi % 4]; gi += 1; per[m][0] += e["dur"]; per[m][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"\n### {f.split('/')[-3]} {f.split('/')[-2]}: {len(ev)/steps:.0f} launches/step, {tot/steps:.0f} us kernel time/step, {span/steps:.0f} us wall/step (first to last kernel)")
print("| kind | launches/step | us/step | % of kernel time |\n|---|---|---|---|")
for k, (us, n) in sorted(agg.items(), key=lambda x: -x[1][0]): print(f"| {k} | {n/steps:.1f} | {us/steps:.0f} | {100*us/tot:.1f} |")
print(f"\nDeepGEMM by layer (by launch order; {gi/steps/layers:.2f} GEMMs per layer per step):")
for m, (us, n) in per.items(): print(f"| {m} | {n/steps:.1f} | {us/steps:.0f} | {100*us/tot:.1f} | {us/n:.1f} us/launch |")
print("\ntop kernel names:")
for (k, n), (us, c) in sorted(names.items(), key=lambda x: -x[1][0])[:14]: print(f"  {100*us/tot:5.1f}%  {c/steps:7.1f}/step  {us/c:8.1f} us  [{k}] {n}")
