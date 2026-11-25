"""Print kernels around the Kth MoE gate-up bmm in step S: start offset, dur, stream. usage: layer_view.py ARM C [K]"""
import glob, gzip, json, sys
arm, C = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 400
f = max(glob.glob(f"{arm}/traces_c{C}/*.json*"), key=lambda p: len(open(p, "rb").read()))
ev = sorted((e["ts"], e["dur"], e["tid"], e["name"]) for e in json.load(gzip.open(f, "rt"))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e)
idx = [i for i, e in enumerate(ev) if e[3].startswith("bmm_E4m3")][K]
t0 = ev[idx][0]
short = lambda n: n.replace("void ", "")[:70]
for ts, d, tid, n in ev[idx - 12: idx + 14]:
    print(f"{ts - t0:+8.1f} {d:6.1f} {'#' if ts==t0 else ' '} s{tid:<5} {short(n)}")
