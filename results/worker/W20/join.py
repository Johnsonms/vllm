"""W20: per MoE layer, which branch reaches the join last (shared expert on the aux stream vs routed experts on the main stream).
usage: join.py TRACE_DIR [label]   (uses the rank-0 trace; kernel events only)
Per layer, anchored on the shared expert's act_and_mul kernel (aux stream):
  routed_end  = end of the MoE finalize kernel (main stream)
  shared_end  = end of the last aux-stream kernel before the join
  join        = start of the first main-stream kernel after the finalize kernel (the add of shared_out)
  gate        = aux kernels between the shared down-proj GEMM and the sigmoid (the (1, 2560) gate GEMM)"""
import bisect, collections, glob, gzip, json, statistics as st, sys
d = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else d
f = [p for p in glob.glob(d + "/*rank0*.json*")][0]
ev = [e for e in json.load((gzip.open if f.endswith("gz") else open)(f, "rt"))["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
ev.sort(key=lambda e: e["ts"])
K = [(e["ts"], e["dur"], e["name"], e["args"].get("stream")) for e in ev]
ts = [k[0] for k in K]
anchors = [i for i, k in enumerate(K) if "act_and_mul" in k[2]]
rows = []; gate_names = collections.Counter(); shapes = collections.Counter()
for a, a_next in zip(anchors, anchors[1:] + [None]):
    t0, _, _, aux = K[a]
    hi = bisect.bisect_left(ts, t0 + 150)
    win = K[max(0, a - 12):hi]
    fin = next((k for k in win if "finalize" in k[2] and k[0] > t0 - 5), None)
    if fin is None: continue
    main = fin[3]; routed_end = fin[0] + fin[1]
    join = next((k for k in win if k[3] == main and k[0] >= routed_end - 0.5 and k is not fin), None)
    if join is None: continue
    auxk = [k for k in win if k[3] == aux and t0 - 12 <= k[0] < join[0] + 1]
    if not auxk: continue
    shared_end = max(k[0] + k[1] for k in auxk)
    sig = next((k for k in auxk if "sigmoid" in k[2]), None)
    act_i = auxk.index(next(k for k in auxk if "act_and_mul" in k[2]))
    if sig is None or len(auxk) < act_i + 3: continue
    down = auxk[act_i + 1]
    gate = [k for k in auxk if down[0] < k[0] < sig[0]]
    if not gate: continue
    for k in gate: gate_names[k[2][:60]] += 1
    shapes[len(auxk)] += 1
    router = next((k for k in win if k[3] == main and t0 - 12 <= k[0] <= t0 + 3), None)
    routed_start = router[0] if router else t0
    period = (K[a_next][0] - t0) if a_next is not None else None
    rows.append(dict(diff=shared_end - routed_end, gate_dur=sum(k[1] for k in gate), gate_span=gate[-1][0] + gate[-1][1] - (down[0] + down[1]),
                     gate_wait=gate[0][0] - (down[0] + down[1]), shared_len=shared_end - auxk[0][0], routed_len=routed_end - routed_start,
                     join_wait=join[0] - routed_end, join_after_shared=join[0] - shared_end, period=period))
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
n = len(rows); diff = [r["diff"] for r in rows]
per = [r["period"] for r in rows if r["period"] is not None and r["period"] < 400]
print(f"== {label}: {n} MoE layers in the window (file {f.split('/')[-1][:40]})")
print(f"shared branch ends last: {sum(x > 0 for x in diff)}/{n} = {100*sum(x > 0 for x in diff)/n:.1f} %")
print(f"shared_end - routed_end (us): p10 {q(diff,.1):+.1f}  p25 {q(diff,.25):+.1f}  median {st.median(diff):+.1f}  p75 {q(diff,.75):+.1f}  p90 {q(diff,.9):+.1f}")
for key, name in (("gate_dur", "gate kernel time"), ("gate_wait", "wait before gate starts"), ("gate_span", "down-proj end -> gate end"), ("shared_len", "shared branch length"),
                  ("routed_len", "routed branch length"), ("join_wait", "routed end -> join start"), ("join_after_shared", "shared end -> join start")):
    v = [r[key] for r in rows]; print(f"{name:28s} median {st.median(v):6.1f}  p25 {q(v,.25):6.1f}  p75 {q(v,.75):6.1f}")
print(f"decoder layer period (act_and_mul to next, < 400 us): median {st.median(per):.1f} us, mean {st.mean(per):.1f} us, n {len(per)}")
print("gate kernels:", dict(gate_names.most_common(4)))
