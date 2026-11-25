"""B5 table: per run median TPOT per concurrency, TTFT c4 control, GSM8K, arm means and deltas vs default.
usage: python3 analyze_b5.py ROOT"""
import glob, json, os, re, statistics as st, sys
root = sys.argv[1]; CS = [32, 96, 128, 192, 256, 384]; ARMS = ["default", "8", "16", "32"]
def j(p, k):
    try: return json.load(open(p))[k]
    except Exception: return None
runs = {}
print("| run | " + " | ".join(f"c{c}" for c in CS) + " | TTFT c4 1k (ms) | GSM8K | AOT loads |")
print("|---|" + "---|" * (len(CS) + 3))
for r in (1, 2):
    for a in ARMS:
        d = f"{root}/r{r}_{a}"
        if not os.path.isdir(d): continue
        t = [j(f"{d}/c{c}.json", "median_tpot_ms") for c in CS]; tt = j(f"{d}/ttft_c4.json", "median_ttft_ms")
        g = j(f"{d}/gsm8k.json", "accuracy"); aot = open(f"{d}/serve.log").read().count("Directly load AOT")
        runs[(r, a)] = (t, tt)
        tts = f"{tt:.1f}" if tt else "-"; gs = f"{g:.3f}" if g else "-"
        print(f"| r{r}_{a} | " + " | ".join(f"{x:.2f}" if x else "-" for x in t) + f" | {tts} | {gs} | {aot} |")
print("\n| arm (mean of 2 rounds) | " + " | ".join(f"c{c}" for c in CS) + " | TTFT c4 |\n|---|" + "---|" * (len(CS) + 1))
mean = {}
for a in ARMS:
    rs = [runs[(r, a)] for r in (1, 2) if (r, a) in runs]
    if not rs: continue
    mean[a] = ([st.mean(x[0][i] for x in rs if x[0][i]) for i in range(len(CS))], st.mean(x[1] for x in rs if x[1]))
    print(f"| {a} | " + " | ".join(f"{v:.2f}" for v in mean[a][0]) + f" | {mean[a][1]:.1f} |")
print("\n| Δ TPOT vs default | " + " | ".join(f"c{c}" for c in CS) + " | TTFT c4 |\n|---|" + "---|" * (len(CS) + 1))
for a in ARMS[1:]:
    if a not in mean: continue
    b = mean["default"]
    print(f"| {a} MB | " + " | ".join(f"{100*(p-q)/q:+.1f}%" for p, q in zip(mean[a][0], b[0])) + f" | {100*(mean[a][1]-b[1])/b[1]:+.1f}% |")
for p in sorted(glob.glob(f"{root}/probe_*/serve.log")):
    s = open(p).read()
    mx = re.findall(r"Flashinfer max size: [^\n]*?Fusion: \d+", s); be = re.findall(r"workspace with backend=\w+", s)
    print(f"\n{os.path.basename(os.path.dirname(p))}: {mx[:1]} {sorted(set(be))} AOT={s.count('Directly load AOT')} tracebacks={s.count('Traceback')}")
