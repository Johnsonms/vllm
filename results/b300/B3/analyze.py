"""Per-run median TPOT, mean per arm, delta, GSM8K. usage: python3 analyze.py DIR [DIR...]"""
import glob, json, os, statistics as st, sys

for root in sys.argv[1:]:
    runs = sorted(glob.glob(os.path.join(root, "r*_*")), key=lambda p: int(os.path.basename(p)[1:].split("_")[0]))
    cs = sorted({int(os.path.basename(f)[1:-5]) for r in runs for f in glob.glob(r + "/c*.json")})
    print(f"\n#### {os.path.basename(root)}\n")
    print("| run | " + " | ".join(f"c{c} TPOT (ms)" for c in cs) + " | GSM8K |")
    print("|---|" + "---|" * (len(cs) + 1))
    arm = {"base": {c: [] for c in cs}, "pr": {c: [] for c in cs}}
    gsm = {"base": [], "pr": []}
    for r in runs:
        name = os.path.basename(r); a = name.split("_")[1]
        row = []
        for c in cs:
            try:
                v = json.load(open(f"{r}/c{c}.json"))["median_tpot_ms"]; arm[a][c].append(v); row.append(f"{v:.3f}")
            except FileNotFoundError:
                row.append("-")
        try:
            g = json.load(open(f"{r}/gsm8k.json"))["accuracy"]; gsm[a].append(g); gs = f"{g:.3f}"
        except Exception:
            gs = "-"
        print(f"| {name} | " + " | ".join(row) + f" | {gs} |")
    mb = [st.mean(arm["base"][c]) if arm["base"][c] else None for c in cs]
    mp = [st.mean(arm["pr"][c]) if arm["pr"][c] else None for c in cs]
    print("| **base mean** | " + " | ".join(f"{x:.3f}" if x else "-" for x in mb) + f" | {st.mean(gsm['base']):.3f} |" if gsm["base"] else "")
    print("| **pr mean** | " + " | ".join(f"{x:.3f}" if x else "-" for x in mp) + f" | {st.mean(gsm['pr']):.3f} |" if gsm["pr"] else "")
    print("| **Δ pr vs base** | " + " | ".join(f"**{100*(p-b)/b:+.1f}%**" if b and p else "-" for b, p in zip(mb, mp)) + " | |")
