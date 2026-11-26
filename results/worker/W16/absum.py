import json, statistics as st, sys, glob
root = sys.argv[1]
for C in (1, 2, 4, 8, 16, 32):
    per = {}
    for arm in "ab":
        for d in sorted(glob.glob(f"{root}/{arm}[0-9]")):
            v = []
            for k in range(1, 5):
                try: v.append(json.load(open(f"{d}/c{C}_k{k}.json"))["median_tpot_ms"])
                except Exception: pass
            if v: per.setdefault(arm, []).append(st.median(v))
    if "a" not in per or "b" not in per: continue
    a, b = per["a"], per["b"]; ma, mb = st.mean(a), st.mean(b)
    print(f"c{C}: base {' '.join(f'{x:.3f}' for x in a)} (spread {100*(max(a)-min(a))/ma:.2f}%) | cand {' '.join(f'{x:.3f}' for x in b)} (spread {100*(max(b)-min(b))/mb:.2f}%) | delta {100*(mb/ma-1):+.2f}%")
