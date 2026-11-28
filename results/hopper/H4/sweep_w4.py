import itertools, json, statistics as st, sys, torch
from flashinfer.testing import bench_gpu_time_with_cupti
from quack.gemm_w4 import gemm_w4a16, prepare_w4_weight
torch.manual_seed(0); dev = "cuda"; G = 128
SHAPES = {"8b.qkv": (4096, 6144), "8b.o": (4096, 4096), "8b.gate_up": (4096, 24576), "8b.down": (12288, 4096),
          "32b.qkv": (5120, 10240), "32b.o": (8192, 5120), "32b.gate_up": (5120, 51200), "32b.down": (25600, 5120)}
def bench(fn, args): return st.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, input_args=args, repeat_time_ms=60)) * 1e3
res = {}
for name, (K, N) in SHAPES.items():
    q = torch.randint(0, 16, (N, K), device=dev, dtype=torch.uint8); s = (torch.rand(N, K // G, device=dev) * .01 + .002).to(torch.bfloat16)
    z = torch.randint(0, 16, (N, K // G), device=dev, dtype=torch.uint8)
    blob, sf = prepare_w4_weight((q[:, 0::2] | (q[:, 1::2] << 4)).contiguous(), (s, z), wformat="int4awq")
    res[name] = {}
    for M in [1, 32, 512, 2048]:
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        auto = bench(lambda x: gemm_w4a16(x, blob, sf, wformat="int4awq"), (x,))
        best = (auto, "auto")
        tns = [16, 32, 64] if M <= 32 else [128, 192, 256]
        for tm, tn, sk in itertools.product([64, 128], tns, [1, 2, 4]):
            if M > 256 and sk > 2: continue
            try:
                gemm_w4a16(x, blob, sf, wformat="int4awq", tile_m=tm, tile_n=tn, split_k=sk); torch.cuda.synchronize()
                t = bench(lambda x: gemm_w4a16(x, blob, sf, wformat="int4awq", tile_m=tm, tile_n=tn, split_k=sk), (x,))
            except Exception as e:
                continue
            if t < best[0]: best = (t, f"tm{tm} tn{tn} sk{sk}")
        res[name][M] = dict(auto_us=auto, best_us=best[0], best_cfg=best[1])
        print(f"{name:12s} M={M:5d}: auto {auto:7.1f}us  best {best[0]:7.1f}us ({best[1]})", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
