"""Top-N GPU kernels from a torch-profiler trace dir, with cuBLAS BF16 GEMM marking.
usage: python -I kernel_table.py <trace_dir> <out.json> [topN]
"""
import glob, gzip, json, re, sys, collections

CUBLAS = re.compile(r"(\bnvjet_|splitKreduce_kernel|\bdot_kernel|reduce_1Block_kernel|cutlass_80_|gemvx?_kernel|gemmk1_kernel)")

def short(n):
    return n if len(n) <= 110 else n[:107] + "..."

files = sorted(glob.glob(sys.argv[1] + "/*.json*"))
assert files, "no trace"
f = max(files, key=lambda p: len(open(p, "rb").read()))  # engine-core rank-0 trace is the large one
op = gzip.open if f.endswith(".gz") else open
ev = json.load(op(f, "rt"))["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
agg = collections.defaultdict(lambda: [0.0, 0])
for e in k:
    a = agg[e["name"]]; a[0] += e["dur"]; a[1] += 1
total = sum(v[0] for v in agg.values())
span = max(e["ts"] + e["dur"] for e in k) - min(e["ts"] for e in k)
rows = sorted(agg.items(), key=lambda x: -x[1][0])
cub_t = sum(v[0] for n, v in agg.items() if CUBLAS.search(n))
cub_n = sum(v[1] for n, v in agg.items() if CUBLAS.search(n))
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 20
out = {
    "trace": f.split("/")[-1], "kernel_launches": len(k), "summed_kernel_us": round(total, 1),
    "gpu_span_us": round(span, 1),
    "cublas_bf16_us": round(cub_t, 1), "cublas_bf16_share_pct": round(100 * cub_t / total, 2),
    "cublas_bf16_launches": cub_n,
    "kernels": [{"name": n, "us": round(v[0], 1), "pct": round(100 * v[0] / total, 2), "calls": v[1],
                 "cublas_bf16": bool(CUBLAS.search(n))} for n, v in rows],
}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"{out['trace']}: {len(k)} launches, summed {total:.0f} us, span {span:.0f} us, "
      f"cuBLAS BF16 {cub_t:.0f} us = {out['cublas_bf16_share_pct']}% ({cub_n} launches)")
print("| # | kernel | us | % | calls | cuBLAS BF16 |\n|---|---|---|---|---|---|")
for i, r in enumerate(out["kernels"][:topn], 1):
    print(f"| {i} | `{short(r['name'])}` | {r['us']:.1f} | {r['pct']:.1f} | {r['calls']} | {'**yes**' if r['cublas_bf16'] else ''} |")
