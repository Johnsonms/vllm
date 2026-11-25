"""GPU-busy union per step (any kernel running) vs summed kernel time -> overlap. usage: union.py TRACE_DIR"""
import glob, gzip, json, sys, statistics as st
f = max(glob.glob(sys.argv[1] + "/*.json*"), key=lambda p: len(open(p, "rb").read()))
ev = sorted(((e["ts"], e["ts"] + e["dur"], e["name"]) for e in json.load(gzip.open(f, "rt"))["traceEvents"]
             if e.get("cat") == "kernel" and "dur" in e))
from collections import Counter
cnt = Counter(n for _, _, n in ev); once = [n for n, c in cnt.items() if c == 32]
first = min(once, key=lambda n: next(s for s, _, m in ev if m == n))
starts = [s for s, _, n in ev if n == first]
U, S = [], []
for a, b in zip(starts, starts[1:]):
    seg = [(s, e) for s, e, _ in ev if a <= s < b]
    u = 0; cur_s, cur_e = seg[0]
    for s, e in seg[1:]:
        if s > cur_e: u += cur_e - cur_s; cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    u += cur_e - cur_s; U.append(u); S.append(sum(e - s for s, e in seg))
print(f"{sys.argv[1]}: busy-union/step median {st.median(U):.0f} us, summed {st.median(S):.0f} us, overlap {st.median(S)-st.median(U):.0f} us")
