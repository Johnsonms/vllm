"""Per-step wall time and per-stream busy time from a torch trace. usage: timeline.py TRACE_DIR"""
import glob, gzip, json, sys, collections, statistics as st
f = max(glob.glob(sys.argv[1] + "/*.json*"), key=lambda p: len(open(p, "rb").read()))
ev = [e for e in json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
ev.sort(key=lambda e: e["ts"])
# step boundary: kernels called exactly 32 times; use the earliest-in-step such name
cnt = collections.Counter(e["name"] for e in ev)
once = [n for n, c in cnt.items() if c == 32]
first = min(once, key=lambda n: next(e["ts"] for e in ev if e["name"] == n))
starts = [e["ts"] for e in ev if e["name"] == first]
steps = [b - a for a, b in zip(starts, starts[1:])]
streams = collections.defaultdict(float)
for e in ev:
    if starts[0] <= e["ts"] < starts[-1]: streams[e["tid"]] += e["dur"]
n = len(steps)
print(f"marker `{first[:60]}`; {n} steps: wall/step median {st.median(steps):.0f} us (min {min(steps):.0f}, max {max(steps):.0f})")
for tid, us in sorted(streams.items(), key=lambda x: -x[1]):
    print(f"  stream {tid}: busy {us/n:.0f} us/step")
# main-stream idle gaps inside steps
main = max(streams, key=streams.get)
m = [e for e in ev if e["tid"] == main and starts[0] <= e["ts"] < starts[-1]]
gap = sum(max(0, b["ts"] - (a["ts"] + a["dur"])) for a, b in zip(m, m[1:]))
print(f"  main stream {main}: idle {gap/n:.0f} us/step")
