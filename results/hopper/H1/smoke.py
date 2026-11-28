"""H1 QuACK smoke test on SM90 (no vLLM)."""
import os, time, traceback, torch, torch.nn.functional as F
os.environ.setdefault("QUACK_PRINT_AUTOTUNING", "0")
import quack
from quack.gemm_interface import gemm, gemm_gated_out
from quack.topk import topk
torch.manual_seed(0)
dev = "cuda"
def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()
def run(name, fn):
    t = time.time()
    try:
        msg = fn(); print(f"PASS {name} ({time.time()-t:.1f}s) {msg}")
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}"); traceback.print_exc()

def captured_configs():
    import quack.autotuner as at
    out = []
    import gc
    for o in gc.get_objects():
        if isinstance(o, at.Autotuner) and o.cache:
            for k, v in o.cache.items(): out.append((o.fn.__name__, str(v)))
    return out

# 1. plain gemm
def t_gemm():
    A = torch.randn(4096, 2048, device=dev, dtype=torch.bfloat16)
    B = torch.randn(2048, 1536, device=dev, dtype=torch.bfloat16)
    out = gemm(A, B); ref = A.float() @ B.float()
    return f"rel_err={rel(out, ref):.2e}"
# 2. gemm_gated_out swiglu concat_layout B, dense
def t_gated_dense():
    M, K, I = 1024, 2048, 768
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(2 * I, K, device=dev, dtype=torch.bfloat16) / K**0.5   # [gate; up]
    h = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    gemm_gated_out(x, W.t(), None, h, activation="swiglu", concat_layout="B")
    y = x.float() @ W.float().t(); ref = F.silu(y[:, :I]) * y[:, I:]
    return f"rel_err={rel(h, ref):.2e}"
# 3. varlen cu_seqlens_m, batched B (E, K, 2I)
def t_gated_varlen():
    E, K, I = 8, 2048, 768
    counts = torch.tensor([0, 17, 300, 5, 1000, 64, 0, 130])
    cu = torch.zeros(E + 1, dtype=torch.int32); cu[1:] = counts.cumsum(0)
    M = int(cu[-1]); cu = cu.to(dev)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * I, K, device=dev, dtype=torch.bfloat16) / K**0.5
    h = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    gemm_gated_out(x, w1.transpose(1, 2), None, h, activation="swiglu", cu_seqlens_m=cu, concat_layout="B")
    ref = torch.empty(M, I, device=dev)
    for e in range(E):
        s, t = int(cu[e]), int(cu[e + 1])
        y = x[s:t].float() @ w1[e].float().t(); ref[s:t] = F.silu(y[:, :I]) * y[:, I:]
    # also varlen plain gemm (fc2)
    w2 = torch.randn(E, K, I, device=dev, dtype=torch.bfloat16) / I**0.5
    o = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    gemm(h, w2.transpose(1, 2), out=o, cu_seqlens_m=cu)
    ref2 = torch.empty(M, K, device=dev)
    for e in range(E):
        s, t = int(cu[e]), int(cu[e + 1]); ref2[s:t] = h[s:t].float() @ w2[e].float().t()
    return f"fc1 rel_err={rel(h, ref):.2e} fc2 rel_err={rel(o, ref2):.2e}"
# 4. topk softmax
def t_topk():
    res = []
    for (E, k) in [(128, 8), (64, 4), (64, 8), (32, 2)]:
        for M in [1, 7, 256, 4096]:
            x = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
            v, i = topk(x, k, softmax=True)
            rv, ri = torch.topk(x.float(), k, dim=-1)
            rv = torch.softmax(rv, -1)
            ids_eq = torch.equal(i.long().sort(-1).values, ri.sort(-1).values)
            res.append(f"E{E}k{k}M{M}:ids_eq={ids_eq},maxerr={(v.float()-rv).abs().max().item():.1e}")
    return " ".join(res)
for n, f in [("gemm", t_gemm), ("gemm_gated_out dense swiglu concat B", t_gated_dense),
             ("gemm_gated_out varlen cu_seqlens_m + varlen gemm", t_gated_varlen), ("topk softmax=True bf16", t_topk)]:
    run(n, f)
print("Autotuner picks:")
for fn, c in captured_configs(): print("  ", fn, c)
