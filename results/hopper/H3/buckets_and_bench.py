"""H3 step 1b/3: pick a <=4-bucket SM90 table from the autotune sweep (shared by both models), then
CUPTI-bench (cold L2, CUDA graph) cuBLAS F.linear + silu_and_mul vs QuACK (bucket table and per-M best),
plus small-M F.linear + 2-D Triton SiLU-mul, and relative error vs FP32."""
import itertools, json, re, statistics as st, sys
import torch, torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti
from vllm import _custom_ops as ops
from vllm.model_executor.layers import kopt_swiglu as ks
from quack.gemm_config import GemmConfig

tune = json.load(open(sys.argv[1]))
MS = [16, 32, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
models = list(tune)


def parse(s):
    g = lambda k: re.search(rf"{k}=(\w+)", s)[1]
    return (int(g("tile_m")), int(g("tile_n")), int(g("cluster_m")), int(g("cluster_n")), g("pingpong") == "True")


best = {m: {M: min(tune[m][str(M)].values()) for M in MS} for m in models}
cfgs = set.intersection(*[set(tune[m][str(M)]) for m in models for M in MS])


def ratio(c, M):
    return max(tune[m][str(M)][c] / best[m][M] for m in models)


def best_cfg(Ms):
    return min(cfgs, key=lambda c: (max(ratio(c, M) for M in Ms), sum(ratio(c, M) for M in Ms)))


def select(mods):
    def ratio(c, M):
        return max(tune[m][str(M)][c] / best[m][M] for m in mods)
    cs = set.intersection(*[set(tune[m][str(M)]) for m in mods for M in MS])
    def bc(Ms):
        return min(cs, key=lambda c: (st.mean(ratio(c, M) for M in Ms), max(ratio(c, M) for M in Ms)))
    sol = None
    for nb in (1, 2, 3, 4):
        for cuts in itertools.combinations(range(1, len(MS)), nb - 1):
            parts = [MS[a:b] for a, b in zip((0,) + cuts, cuts + (len(MS),))]
            chosen = [bc(p) for p in parts]
            rs = [ratio(c, M) for c, p in zip(chosen, parts) for M in p]
            key = (st.mean(rs), max(rs))
            if sol is None or key < sol[0]:
                sol = (key, parts, chosen, rs)
    (mean, worst), parts, chosen, rs = sol
    table = []
    for i, (p, c) in enumerate(zip(parts, chosen)):
        tm, tn, cm, cn, pp = parse(c)
        table.append([p[-1] if i < len(parts) - 1 else 1 << 30, tm, tn, cm, cn, pp])
    print(f"table for {mods} (mean {100*(mean-1):.2f}% / worst {100*(worst-1):.1f}% over per-M autotune):")
    for r, p in zip(table, parts):
        print(f"  M<={r[0] if r[0] < 1<<30 else 'inf'} (sweep Ms {p}): tile {r[1]}x{r[2]} cluster {r[3]}x{r[4]} pingpong={r[5]}")
    return table, mean, worst


SHAPE_KEY = {"qwen3_8b": "24576x4096", "qwen3_32b": "51200x5120"}
tables = {"default": select(models)}
for m in models:
    tables[SHAPE_KEY[m]] = select([m])
table = {k: v[0] for k, v in tables.items()}
worst = {k: v[2] for k, v in tables.items()}; mean = {k: v[1] for k, v in tables.items()}
json.dump(table, open("sm90_table.json", "w"))
ks.SM90_TABLE = {k: [tuple(r) for r in v] for k, v in table.items()}; ks._cfgs = None

SHAPES = {"qwen3_8b": (24576, 4096), "qwen3_32b": (51200, 5120)}
ROUNDS = 5


def bench(fn, args):
    return st.median(st.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True,
                                                         input_args=args, repeat_time_ms=100)) for _ in range(ROUNDS)) * 1e3


def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()


@torch.compile(dynamic=True)
def native_silu_mul(gu):
    d = gu.shape[-1] // 2
    return F.silu(gu[..., :d]) * gu[..., d:]


out = {"table": table, "worst_vs_autotune": worst, "mean_vs_autotune": mean, "rows": []}
torch.manual_seed(0)
for name, (N2, K) in SHAPES.items():
    I = N2 // 2
    W = torch.randn(N2, K, device="cuda", dtype=torch.bfloat16) / K**0.5
    ks._warmup(W)
    for M in [1, 2, 4, 8] + MS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        o = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)

        def unfused(x, o):
            torch.ops._C.silu_and_mul(o, F.linear(x, W))

        def unfused_ind(x, o):
            o.copy_(native_silu_mul(F.linear(x, W)))

        def fused(x, o):
            if M < 16:
                ks.silu_mul_2d(F.linear(x, W), o)
            else:
                ks._quack(x, W, o, ks.pick_config(M, W))
        t_u = bench(unfused, (x, o))
        t_ind = bench(unfused_ind, (x, o))
        t_f = bench(fused, (x, o))
        row = dict(model=name, M=M, cublas_silu_us=t_u, cublas_inductor_silu_us=t_ind, kopt_us=t_f, speedup=t_u / t_f, speedup_vs_inductor=t_ind / t_f)
        if M >= 16:
            bc = GemmConfig(*[], **dict(zip(("tile_m", "tile_n", "cluster_m", "cluster_n", "pingpong"),
                                            parse(min(tune[name][str(M)], key=tune[name][str(M)].get)))),
                            is_dynamic_persistent=False, swap_ab=False, device_capacity=9)
            row["autotuned_us"] = bench(lambda x, o: ks._quack(x, W, o, bc), (x, o))
            row["speedup_autotuned"] = t_u / row["autotuned_us"]
        # accuracy vs FP32 reference
        y = x.float() @ W.float().t(); ref = F.silu(y[:, :I]) * y[:, I:]
        fused(x, o); e_f = rel(o, ref)
        o2 = torch.empty_like(o); unfused(x, o2); e_u = rel(o2, ref)
        row.update(relerr_fused=e_f, relerr_unfused=e_u)
        out["rows"].append(row)
        print(f"{name} M={M}: cuBLAS+silu_and_mul {t_u:.1f}us  cuBLAS+inductor {t_ind:.1f}us  kopt {t_f:.1f}us ({t_u/t_f:.2f}x)"
              + (f"  per-M autotuned {row['autotuned_us']:.1f}us ({row['speedup_autotuned']:.2f}x)" if M >= 16 else "")
              + f"  relerr fused {e_f:.4f} unfused {e_u:.4f}", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
