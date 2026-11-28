"""GPU kernel time share from a torch profiler trace. usage: share.py TRACE.json.gz"""
import gzip, json, re, sys, collections
d = json.load(gzip.open(sys.argv[1]))
k = [e for e in d["traceEvents"] if e.get("cat") == "kernel"]
tot = sum(e["dur"] for e in k)
cat = collections.Counter(); names = collections.Counter(); cnt = collections.Counter()
def c(n):
    l = n.lower()
    if "machete" in l: return "W4 GEMM (machete)"
    if "marlin" in l: return "W4 GEMM (marlin)"
    if "flash" in l or "attn" in l or "attention" in l: return "attention"
    if "gemm" in l or "cutlass" in l or "sm90_xmma" in l or "nvjet" in l: return "other GEMM (bf16/fp16)"
    if "norm" in l: return "norm"
    return "other"
for e in k:
    cat[c(e["name"])] += e["dur"]; names[e["name"][:110]] += e["dur"]; cnt[e["name"][:110]] += 1
print(f"total kernel time {tot/1e3:.1f} ms over {len(k)} kernels")
for n, v in cat.most_common(): print(f"  {100*v/tot:5.1f}%  {n}")
print("top kernels:")
for n, v in names.most_common(8): print(f"  {100*v/tot:5.1f}%  x{cnt[n]:5d}  {n}")
