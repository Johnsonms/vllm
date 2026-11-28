"""H3 step 1a: QuACK gated (swiglu, concat B) autotune sweep on SM90; dumps every config's timing per M."""
import gc, json, sys
import torch
from quack.gemm_interface import gemm_gated_out
import quack.autotuner as at

SHAPES = {"qwen3_8b": (24576, 4096), "qwen3_32b": (51200, 5120)}  # (2I, K)
MS = [16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
res = {}
for name, (N2, K) in SHAPES.items():
    W = torch.randn(N2, K, device="cuda", dtype=torch.bfloat16) / K**0.5
    res[name] = {}
    for M in MS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N2 // 2, device="cuda", dtype=torch.bfloat16)
        gemm_gated_out(x, W.t(), None, out, activation="swiglu", concat_layout="B")
        torch.cuda.synchronize()
        tuner = [o for o in gc.get_objects() if isinstance(o, at.Autotuner) and o.fn.__name__.startswith("mod_linear_act")][0]
        t = {str(c.kwargs["config"]): v[0] for c, v in tuner.configs_timings.items()}
        res[name][M] = t
        b = min(t, key=t.get)
        print(f"{name} M={M}: best {t[b]*1e3:.1f}us {b}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=0)
