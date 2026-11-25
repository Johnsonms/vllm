"""B7 paired A/B table: per pass and c, mean TPOT p50 of 6 bursts per arm, per-pair deltas, mean of passes.
Also GSM8K and greedy-token identity. usage: analyze_paired.py ROOT"""
import json, os, statistics as st, sys
root = sys.argv[1]; CS = [1, 2, 4, 8, 16]
def tp(p): return json.load(open(p))["median_tpot_ms"]
rows = []; passes = [p for p in (1, 2) if os.path.isdir(f"{root}/pass{p}")]
print("| c | " + " | ".join(f"pass {p} (T on GPU {p-1}): T / P, Δ (pair range)" for p in passes) + " | mean Δ of passes |")
print("|---|" + "---|" * (len(passes) + 1))
for c in CS:
    cells, ds = [], []
    for p in passes:
        try:
            t = [tp(f"{root}/pass{p}/T/c{c}_r{r}.json") for r in range(1, 7)]
            q = [tp(f"{root}/pass{p}/P/c{c}_r{r}.json") for r in range(1, 7)]
        except FileNotFoundError:
            cells.append("-"); continue
        pr = [100 * (b - a) / a for a, b in zip(t, q)]; d = 100 * (st.mean(q) - st.mean(t)) / st.mean(t); ds.append(d)
        cells.append(f"{st.mean(t):.3f} / {st.mean(q):.3f}, **{d:+.2f}%** ({min(pr):+.2f} to {max(pr):+.2f}; {sum(x < 0 for x in pr)}/6 neg)")
    print(f"| {c} | " + " | ".join(cells) + f" | **{st.mean(ds):+.2f}%** |" if ds else f"| {c} | " + " | ".join(cells) + " | - |")
print()
for p in passes:
    g = {}
    for a in ("T", "P"):
        try: g[a] = json.load(open(f"{root}/pass{p}/{a}/gsm8k.json"))["accuracy"]
        except Exception: g[a] = None
    try:
        gt, gp = (json.load(open(f"{root}/pass{p}/greedy_{a}.json")) for a in ("T", "P"))
        same = sum(x["tokens"] == y["tokens"] for x, y in zip(gt, gp))
    except Exception: same = None
    print(f"pass {p}: GSM8K T {g['T']} P {g['P']}; greedy tokens identical on {same}/4 prompts (64 tokens each)")
