"""B8: paired TPOT table, GSM8K per run, per-question flips (base->PR and base->base A/A).
usage: analyze_b8.py ROOT"""
import glob, itertools, json, os, statistics as st, sys
root = sys.argv[1]
def tp(p): return json.load(open(p))["median_tpot_ms"]
print("| c | " + " | ".join(f"pass {p} (base on GPU {p-1}): base / PR ms, Δ (pairs)" for p in (1, 2)) + " | **mean Δ** |\n|---|---|---|---|")
for c in (1, 2, 4, 8):
    cells, ds = [], []
    for p in (1, 2):
        a = [tp(f"{root}/pass{p}/base/c{c}_r{r}.json") for r in range(1, 7)]
        b = [tp(f"{root}/pass{p}/pr/c{c}_r{r}.json") for r in range(1, 7)]
        pr = [100 * (y - x) / x for x, y in zip(a, b)]; d = 100 * (st.mean(b) - st.mean(a)) / st.mean(a); ds.append(d)
        cells.append(f"{st.mean(a):.3f} / {st.mean(b):.3f}, {d:+.2f}% ({min(pr):+.2f}..{max(pr):+.2f}; {sum(x < 0 for x in pr)}/6 neg)")
    print(f"| {c} | " + " | ".join(cells) + f" | **{st.mean(ds):+.2f}%** |")
runs = {a: [] for a in ("base", "pr")}
for a in runs:
    for p in (1, 2):
        for g in (1, 2, 3, 4):
            f = f"{root}/pass{p}/{a}/gsm8k_g{g}_perq.json"
            if os.path.exists(f):
                d = json.load(open(f)); runs[a].append((f"p{p}g{g}", d["correct"], d["invalid"]))
print("\n| run | base acc | PR acc | base invalid | PR invalid |\n|---|---|---|---|---|")
for (nb, cb, ib), (np_, cp, ip) in zip(runs["base"], runs["pr"]):
    print(f"| {nb} | {sum(cb)/len(cb):.4f} | {sum(cp)/len(cp):.4f} | {sum(ib)} | {sum(ip)} |")
mb = st.mean(sum(c) / len(c) for _, c, _ in runs["base"]); mp = st.mean(sum(c) / len(c) for _, c, _ in runs["pr"])
sb = st.stdev(sum(c) / len(c) for _, c, _ in runs["base"]); sp = st.stdev(sum(c) / len(c) for _, c, _ in runs["pr"])
print(f"| **mean (sd)** | **{mb:.4f}** ({sb:.4f}) | **{mp:.4f}** ({sp:.4f}) | | |")
def flips(x, y):
    rw = sum(1 for u, v in zip(x, y) if u == 1 and v == 0); wr = sum(1 for u, v in zip(x, y) if u == 0 and v == 1)
    return rw, wr
bp = [flips(cb, cp) for (_, cb, _), (_, cp, _) in itertools.product(runs["base"], runs["pr"])]
bb = [flips(x[1], y[1]) for x, y in itertools.combinations(runs["base"], 2)]
pp = [flips(x[1], y[1]) for x, y in itertools.combinations(runs["pr"], 2)]
def summ(v): return f"right→wrong {st.mean(a for a, _ in v):.1f}, wrong→right {st.mean(b for _, b in v):.1f}, total {st.mean(a + b for a, b in v):.1f} (range {min(a + b for a, b in v)}–{max(a + b for a, b in v)}), over {len(v)} pairs"
print(f"\nflips base→PR (all {len(bp)} run pairs): {summ(bp)}")
print(f"flips base→base (A/A noise floor): {summ(bb)}")
print(f"flips PR→PR (A/A): {summ(pp)}")
nq = len(runs["base"][0][1])
cb = [sum(c[i] for _, c, _ in runs["base"]) for i in range(nq)]; cp = [sum(c[i] for _, c, _ in runs["pr"]) for i in range(nq)]
n = len(runs["base"])
print(f"questions always right: base {sum(x == n for x in cb)}, PR {sum(x == n for x in cp)}; always wrong: base {sum(x == 0 for x in cb)}, PR {sum(x == 0 for x in cp)}; unstable (mixed in either arm): {sum(1 for x, y in zip(cb, cp) if 0 < x < n or 0 < y < n)}")
print(f"questions right in all base runs but wrong in all PR runs: {sum(1 for x, y in zip(cb, cp) if x == n and y == 0)}; reverse: {sum(1 for x, y in zip(cb, cp) if x == 0 and y == n)}")
