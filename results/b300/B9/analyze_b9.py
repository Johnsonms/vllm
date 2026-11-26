import json, glob, math, random, statistics as st, sys
root = sys.argv[1] if len(sys.argv) > 1 else "gsm"
arms = {}
for a in "ABC":
    runs = {}
    for f in sorted(glob.glob(f"{root}/B9_[XY]/{a}/s*_perq.json")):
        s = int(f.split("/s")[-1].split("_")[0]); d = json.load(open(f))
        runs[s] = d
    arms[a] = runs
name = {"A": "main fb159244e6", "B": "v1 c5ef00f633", "C": "v2 152bca1e65"}
seeds = sorted(arms["A"])
acc = {a: [sum(arms[a][s]["correct"]) / len(arms[a][s]["correct"]) for s in seeds] for a in "ABC"}
inv = {a: sum(sum(arms[a][s]["invalid"]) for s in seeds) for a in "ABC"}
nq = len(arms["A"][seeds[0]]["correct"])
print(f"questions {nq}, seeds {seeds}")
print("| arm | " + " | ".join(f"s{s}" for s in seeds) + " | mean | sd | invalid (total) |")
print("|---|" + "---|" * len(seeds) + "---|---|---|")
for a in "ABC":
    print(f"| {name[a]} | " + " | ".join(f"{x:.4f}" for x in acc[a]) + f" | **{st.mean(acc[a]):.4f}** | {st.stdev(acc[a]):.4f} | {inv[a]} |")
def welch(x, y):
    vx, vy = st.variance(x) / len(x), st.variance(y) / len(y)
    return (st.mean(y) - st.mean(x)) / math.sqrt(vx + vy)
def paired(x, y):
    d = [b - a for a, b in zip(x, y)]
    return st.mean(d) / (st.stdev(d) / math.sqrt(len(d)))
cnt = {a: [sum(arms[a][s]["correct"][q] for s in seeds) for q in range(nq)] for a in "ABC"}
def ge4(c1, c2): return sum(1 for q in range(nq) if abs(c1[q] - c2[q]) >= 4)
def directional(c1, c2): return sum(1 for q in range(nq) if c2[q] - c1[q] >= 4), sum(1 for q in range(nq) if c1[q] - c2[q] >= 4)
def null(a1, a2, iters=2000, rng=random.Random(0)):
    pool = [arms[a1][s]["correct"] for s in seeds] + [arms[a2][s]["correct"] for s in seeds]
    out = []
    for _ in range(iters):
        idx = list(range(16)); rng.shuffle(idx)
        c1 = [sum(pool[i][q] for i in idx[:8]) for q in range(nq)]
        c2 = [sum(pool[i][q] for i in idx[8:]) for q in range(nq)]
        out.append(ge4(c1, c2))
    out.sort(); return st.mean(out), out[int(0.95 * len(out))]
print()
for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
    up, down = directional(cnt[a], cnt[b])
    m, p95 = null(a, b)
    print(f"{name[b].split()[0]} vs {name[a].split()[0]}: Δ {st.mean(acc[b]) - st.mean(acc[a]):+.4f} "
          f"({(st.mean(acc[b]) - st.mean(acc[a])) * 100:+.2f} pp), Welch t {welch(acc[a], acc[b]):+.2f}, paired-by-seed t {paired(acc[a], acc[b]):+.2f}; "
          f"questions differing by >=4/8: {up + down} ({up} better in {name[b].split()[0]}, {down} worse); "
          f"label-permutation null: mean {m:.1f}, 95th pct {p95}")
