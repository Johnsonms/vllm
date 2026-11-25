"""cuBLAS kernels F.linear picks on this GPU for the Qwen3.6-35B-A3B TP1 BF16 shapes at M=1..8.
Times are torch-profiler kernel durations (eager, warm L2, 50 calls, median per kernel name).
usage: python cublas_names.py OUT.json"""
import json, statistics as st, sys, torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
SHAPES = {"hc_up (10240,320)": (10240,320), "shared_gate (1,2560)": (1,2560), "shared_down (2560,160)": (2560,160), "router (512,2560)": (512,2560), "shared_gate_up (320,2560)": (320,2560), "hc_down_inject (336,10240)": (336,10240), "hc_mixer_down (320,10240)": (320,10240), "gdn_out (2560,1536)": (2560,1536), "ple_kv (12800,2560)": (12800,2560), "lm_head (62080,2560)": (62080,2560)}
out = []
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for m in (1, 4, 8, 16, 32, 128):
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
