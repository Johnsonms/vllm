"""Per-trace check for `vllm bench forward --profile-dir` traces.
usage: python trace_tail.py <dir with *.trace.json.gz>
Prints, per case and rank: GPU activity count, count % 3, how far the last GPU activity ends after the last
host cudaDeviceSynchronize returned, and the margin left before the profiler's "Record Window End".
Ranks of one case should have equal counts; a lower count with a margin near 0 means the tail was cut off.
"""
import glob, gzip, json, os, sys
G = {"kernel", "gpu_memcpy", "gpu_memset"}
print("case.rank        n  n%3  lastGPUend-syncEnd(us)  windowEnd-lastGPUend(us)")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.trace.json.gz"))):
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    acts = [e for e in ev if e.get("cat") in G and "dur" in e]
    last = max(e["ts"] + e["dur"] for e in acts)
    stop = [e for e in ev if e.get("name") == "cudaProfilerStop"][0]["ts"]
    sync = max(e["ts"] + e["dur"] for e in ev if e.get("name") == "cudaDeviceSynchronize" and e["ts"] < stop)
    we = [e for e in ev if e.get("name") == "Record Window End"][0]["ts"]
    print(os.path.basename(f)[:-14].ljust(14), len(acts), len(acts) % 3, round(last - sync, 1), round(we - last, 1))
