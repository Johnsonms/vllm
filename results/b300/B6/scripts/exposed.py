"""Exposed (sole-running) kernel time per step, by name; base vs pr at one C. usage: exposed.py C [N]"""
import glob, gzip, json, sys, collections
C = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
def load(arm):
    f = max(glob.glob(f"{arm}/traces_c{C}/*.json*"), key=lambda p: len(open(p, "rb").read()))
    ev = [(e["ts"], e["ts"] + e["dur"], e["name"][:90]) for e in json.load(gzip.open(f, "rt"))["traceEvents"]
          if e.get("cat") == "kernel" and "dur" in e]
    pts = sorted([(s, 1, i) for i, (s, e, n) in enumerate(ev)] + [(e, -1, i) for i, (s, e, n) in enumerate(ev)])
    active = set(); last = None; exp = collections.Counter(); both = 0.0
    for t, typ, i in pts:
        if last is not None and len(active) == 1: exp[ev[next(iter(active))][2]] += t - last
        if last is not None and len(active) >= 2: both += t - last
        active.add(i) if typ == 1 else active.discard(i); last = t
    return {k: v / 32 for k, v in exp.items()}, both / 32
b, bb = load("base"); p, pb = load("pr")
print(f"c{C}: exposed total base {sum(b.values()):.0f} pr {sum(p.values()):.0f} us/step; overlapped (>=2 running) base {bb:.0f} pr {pb:.0f}")
print("| kernel | exposed base µs/step | exposed pr | Δ |\n|---|---|---|---|")
for k in sorted(set(b) | set(p), key=lambda k: -abs(p.get(k, 0) - b.get(k, 0)))[:N]:
    print(f"| `{k}` | {b.get(k,0):.0f} | {p.get(k,0):.0f} | {p.get(k,0)-b.get(k,0):+.0f} |")
