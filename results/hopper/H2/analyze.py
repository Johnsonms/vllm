"""Paired A/B table. usage: analyze.py ROOT [label]
Per kind/c: pass means (base -> PR, Δ), mean of the two pass deltas, PR faster in k/12 pairs; GSM8K; greedy match."""
import glob, json, os, statistics as st, sys

root = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(root)
METRIC = {"ttft": "median_ttft_ms", "tpot": "median_tpot_ms"}
print(f"### {label}\n")
for kind in ("ttft", "tpot"):
    cs = sorted({int(os.path.basename(f).split("_c")[1].split("_")[0])
                 for f in glob.glob(f"{root}/pass*/base/{kind}_c*_r*.json")})
    if not cs:
        continue
    unit = "ms"
    print(f"| {kind.upper()} c | pass 1 base → PR | pass 2 base → PR | per-round Δ (12) | **mean Δ** | PR faster |")
    print("|---:|---|---|---|---:|---:|")
    for c in cs:
        cells, ds, rounds = [], [], []
        for p in (1, 2):
            a = [json.load(open(f))[METRIC[kind]] for f in sorted(glob.glob(f"{root}/pass{p}/base/{kind}_c{c}_r*.json"))]
            b = [json.load(open(f))[METRIC[kind]] for f in sorted(glob.glob(f"{root}/pass{p}/pr/{kind}_c{c}_r*.json"))]
            if not a or len(a) != len(b):
                cells.append("n/a"); continue
            rounds += [100 * (y - x) / x for x, y in zip(a, b)]
            d = 100 * (st.mean(b) - st.mean(a)) / st.mean(a); ds.append(d)
            cells.append(f"{st.mean(a):.3f} → {st.mean(b):.3f} ({d:+.2f}%)")
        rr = ", ".join(f"{x:+.1f}" for x in rounds)
        mean = f"**{st.mean(ds):+.2f}%**" if ds else "n/a"
        print(f"| {c} | {cells[0]} | {cells[1] if len(cells) > 1 else 'n/a'} | {rr} | {mean} | {sum(x < 0 for x in rounds)}/{len(rounds)} |")
    print()
for arm in ("base", "pr"):
    f = f"{root}/pass1/{arm}/gsm8k.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"GSM8K {arm}: {d['accuracy']:.4f} (invalid {d.get('invalid_rate', 0):.3f})")
for p in (1, 2):
    fb, fp = f"{root}/pass{p}/greedy_base.json", f"{root}/pass{p}/greedy_pr.json"
    if os.path.exists(fb) and os.path.exists(fp):
        gb, gp = json.load(open(fb)), json.load(open(fp))
        same, div = 0, []
        for x, y in zip(gb, gp):
            if x["tokens"] == y["tokens"]:
                same += 1
            else:
                i = next(i for i, (u, v) in enumerate(zip(x["tokens"], y["tokens"])) if u != v) if any(u != v for u, v in zip(x["tokens"], y["tokens"])) else min(len(x["tokens"]), len(y["tokens"]))
                div.append(i)
        print(f"greedy pass{p}: {same}/{len(gb)} identical over 64 tokens" + (f"; first divergence at tokens {div}" if div else ""))
    aot = [open(f).read().strip() for f in glob.glob(f"{root}/pass{p}/*/checks.txt")]
    if aot:
        print(f"pass{p} {'; '.join(aot)}")
