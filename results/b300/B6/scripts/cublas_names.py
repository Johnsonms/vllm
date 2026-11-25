"""cuBLAS kernels F.linear picks on this GPU for the Qwen3.6-35B-A3B TP1 BF16 shapes at M=1..8.
Times are torch-profiler kernel durations (eager, warm L2, 50 calls, median per kernel name).
usage: python cublas_names.py OUT.json"""
import json, statistics as st, sys, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
SHAPES = {"in_proj_ba (64,2048)": (64, 2048), "shared_gate (1,2048)": (1, 2048),
          "router (256,2048)": (256, 2048), "lm_head (248320,2048)": (248320, 2048)}
out = []
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for m in (1, 2, 3, 4, 5, 6, 8):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        for _ in range(10): F.linear(x, w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(50): F.linear(x, w)
            torch.cuda.synchronize()
        ks = {}
        for e in p.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                ks.setdefault(e.name, []).append(e.device_time)
        rec = dict(shape=name, n=n, k=k, m=m, kernels=[dict(name=kn, calls_per_linear=len(v) / 50, median_us=round(st.median(v), 2)) for kn, v in ks.items()])
        rec["total_us"] = round(sum(x["calls_per_linear"] * x["median_us"] for x in rec["kernels"]), 2)
        out.append(rec)
        print(f"{name} M={m}: {rec['total_us']:.2f} us  " + " + ".join(f"{x['name'][:70]} ({x['median_us']})" for x in rec["kernels"]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
