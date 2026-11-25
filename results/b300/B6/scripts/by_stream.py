"""kernel us/step per stream, base vs pr at one C. usage: by_stream.py C [N]"""
import glob, gzip, json, sys, collections
C = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
def load(arm):
    f = max(glob.glob(f"{arm}/traces_c{C}/*.json*"), key=lambda p: len(open(p, "rb").read()))
    ev = [e for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
    tot = collections.Counter(e["tid"] for e in ev for _ in [0])
    busy = collections.defaultdict(float)
    for e in ev: busy[e["tid"]] += e["dur"]
    main = max(busy, key=lambda t: tot[t])  # stream with most launches
    d = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        k = ("main" if e["tid"] == main else "aux", e["name"][:90]); d[k][0] += e["dur"] / 32; d[k][1] += 1 / 32
    return d, {("main" if t == main else "aux"): (busy[t] / 32, tot[t] / 32) for t in busy}
b, sb = load("base"); p, sp = load("pr")
print("streams base:", {k: (round(v[0]), round(v[1])) for k, v in sb.items()}, " pr:", {k: (round(v[0]), round(v[1])) for k, v in sp.items()})
keys = sorted(set(b) | set(p), key=lambda k: -abs(p.get(k, [0])[0] - b.get(k, [0])[0]))
print("| stream | kernel | base µs/step (calls) | pr µs/step (calls) | Δ |\n|---|---|---|---|---|")
for k in keys[:N]:
    bv, pv = b.get(k, [0, 0]), p.get(k, [0, 0])
    print(f"| {k[0]} | `{k[1]}` | {bv[0]:.0f} ({bv[1]:g}) | {pv[0]:.0f} ({pv[1]:g}) | {pv[0]-bv[0]:+.0f} |")
