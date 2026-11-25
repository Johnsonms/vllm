"""Per-layer medians relative to the routed gate-up bmm start. usage: layers.py ARM C"""
import glob, gzip, json, sys, statistics as st, bisect
arm, C = sys.argv[1], sys.argv[2]
f = max(glob.glob(f"/scratch/kopt-w13/prof/{arm}/traces_c{C}/*.json*"), key=lambda p: len(open(p, "rb").read()))
ev = sorted((e["ts"], e["dur"], e["name"]) for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e)
def pick(sub): return [(t, d) for t, d, n in ev if sub in n]
g = pick("bmm_E4m3"); act = pick("act_and_mul"); fin = pick("finalizeKernel"); mul = pick("MulFunctor"); dn = pick("bmm_Bfloat16")
def nxt(lst, t0, t1):
    i = bisect.bisect_left(lst, (t0 - 15, 0)); r = [x for x in lst[i:i+3] if t0 - 15 <= x[0] < t1]; return r[0] if r else None
rows = []
for (t0, d0), (t1, _) in zip(g, g[1:]):
    if t1 - t0 > 400: continue          # step boundary
    a, f_, m, d = nxt(act, t0, t1), nxt(fin, t0, t1), nxt(mul, t0, t1), nxt(dn, t0, t1)
    if None in (a, f_, m, d): continue
    rows.append((a[0]-t0, m[0]+m[1]-t0, f_[0]+f_[1]-t0, d0, d[1], t1-t0, (m[0]+m[1])-(f_[0]+f_[1])))
med = [st.median(c) for c in zip(*rows)]
print(f"{arm:5} c{C} layers {len(rows):5} | act start {med[0]:+6.1f} | shared end {med[1]:6.1f} | routed end {med[2]:6.1f} | gate-up bmm {med[3]:5.1f} | down bmm {med[4]:5.1f} | layer period {med[5]:6.1f} | shared-routed lag {med[6]:+6.1f}")
