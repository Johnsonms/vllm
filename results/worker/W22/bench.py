"""W22: fold residual-add + RMSNorm into QuACK GEMM epilogues (Qwen3 dense BF16 shapes, B200).
Chain 1 (attention out -> MLP in):  o_proj -> add residual -> RMSNorm -> gate_up -> silu*mul
Chain 2 (MLP out -> next attention): down_proj -> add residual -> RMSNorm -> qkv_proj
Arms:  base   = cuBLAS GEMM + Inductor fused add+rmsnorm + cuBLAS GEMM (+ Inductor silu*mul)        [what vLLM runs at M >= 16]
       gated  = QuACK gemm + Inductor add+rmsnorm + QuACK gated GEMM (chain 1 only; isolates the gated-GEMM gain)
       fold   = QuACK gemm_rms (residual + sq-sum in the epilogue, norm weight applied, rstd finalize kernel)
                + QuACK gemm_norm_act (rstd applied per row in the epilogue [+ SwiGLU])
CUDA graph, cold L2, CUPTI, median us. Error vs an fp32 reference of the whole chain. Random bf16 weights.
usage: bench.py 8b|32b OUT.jsonl [Ms]"""
import json, sys, torch, torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti
from quack import gemm_interface as gi
SH = {"8b": dict(H=4096, I=12288, QKV=6144, OIN=4096), "32b": dict(H=5120, I=25600, QKV=10240, OIN=8192)}[sys.argv[1]]
outp = sys.argv[2]; Ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [16, 32, 64, 128, 512, 2048, 8192]
dev = "cuda"; bf = torch.bfloat16; eps = 1e-6; H, I, QKV, OIN = SH["H"], SH["I"], SH["QKV"], SH["OIN"]
torch.manual_seed(0)
def rw(n, k): return (torch.randn(n, k, device=dev) / k**0.5).to(bf)
Wo, Wgu, Wdn, Wqkv = rw(H, OIN), rw(2 * I, H), rw(H, I), rw(QKV, H)
wn = (torch.rand(H, device=dev) + 0.5).to(bf)
# QuACK operands: B is (K, N). .T of an (N, K) weight is a free view (K-major rows stay contiguous).
WoT, WdnT, WqkvT = Wo.T, Wdn.T, Wqkv.T
WguI_T = gi._concat_interleave(Wgu).contiguous().T          # gate/up interleaved along N (offline re-layout)
def t(fn): return round(1e3 * float(torch.tensor(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=60)).median()), 2)
def rel(a, b): return round(((a.float() - b).norm() / b.norm()).item(), 5)
def addnorm(o, res, w):          # vLLM fused_add_rms_norm semantics; Inductor turns this into one Triton reduction kernel
    h = (o.float() + res.float())
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(bf), h.to(bf)
def silu_mul(o): return F.silu(o[:, :I]) * o[:, I:]
out = open(outp, "a")
for M in Ms:
    torch._dynamo.reset(); c_addnorm = torch.compile(addnorm, dynamic=False); c_silu = torch.compile(silu_mul, dynamic=False)
    torch.manual_seed(M)
    a_in = torch.randn(M, OIN, device=dev, dtype=bf); res = torch.randn(M, H, device=dev, dtype=bf)
    m_in = torch.randn(M, I, device=dev, dtype=bf)
    rec = dict(model=sys.argv[1], M=M)
    def step(name, fn, check=None):
        try:
            y = fn(); torch.cuda.synchronize()
            if check is not None: rec.update({f"{name}_{k}": v for k, v in check(y).items()})
            rec[name + "_us"] = t(fn)
        except Exception as e: rec[name + "_fail"] = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][-200:]}"
    # ---------------- chain 1: o_proj -> add+norm -> gate_up -> silu*mul
    hf = a_in.float() @ Wo.float().T + res.float(); yf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps) * wn.float()
    pf = yf @ Wgu.float().T; actf = F.silu(pf[:, :I]) * pf[:, I:]
    chk1 = lambda r: dict(err_act=rel(r[0], actf), err_h=rel(r[1], hf))
    def base1():
        y, h = c_addnorm(F.linear(a_in, Wo), res, wn); return c_silu(F.linear(y, Wgu)), h
    def gated1():
        y, h = c_addnorm(gi.gemm(a_in, WoT), res, wn); return gi.gemm_act(y, WguI_T, activation="swiglu", store_preact=False)[1], h
    def fold1():
        h = torch.empty(M, H, dtype=bf, device=dev)
        d, rstd = gi.gemm_rms(a_in, WoT, C=res, norm_weight=wn, premult_out=h, eps=eps)
        return gi.gemm_norm_act(d, WguI_T, rstd=rstd, activation="swiglu")[1], h
    for f in (base1, gated1, fold1): f(); f()
    step("c1_base", base1, chk1); step("c1_gated", gated1, chk1); step("c1_fold", fold1, chk1)
    # pieces
    o1 = F.linear(a_in, Wo); y1, _ = c_addnorm(o1, res, wn); g1 = F.linear(y1, Wgu)
    step("p_cublas_o", lambda: F.linear(a_in, Wo)); step("p_addnorm", lambda: c_addnorm(o1, res, wn))
    step("p_cublas_gu", lambda: F.linear(y1, Wgu)); step("p_silu", lambda: c_silu(g1))
    step("p_q_o", lambda: gi.gemm(a_in, WoT)); step("p_q_gated", lambda: gi.gemm_act(y1, WguI_T, activation="swiglu", store_preact=False)[1])
    hb = torch.empty(M, H, dtype=bf, device=dev)
    step("p_q_o_rms", lambda: gi.gemm_rms(a_in, WoT, C=res, norm_weight=wn, premult_out=hb, eps=eps))
    d1, r1 = gi.gemm_rms(a_in, WoT, C=res, norm_weight=wn, premult_out=hb, eps=eps)
    step("p_q_norm_gated", lambda: gi.gemm_norm_act(d1, WguI_T, rstd=r1, activation="swiglu")[1])
    # ---------------- chain 2: down_proj -> add+norm -> qkv
    hf2 = m_in.float() @ Wdn.float().T + res.float(); yf2 = hf2 * torch.rsqrt(hf2.pow(2).mean(-1, keepdim=True) + eps) * wn.float()
    qf = yf2 @ Wqkv.float().T
    chk2 = lambda r: dict(err_qkv=rel(r[0], qf), err_h=rel(r[1], hf2))
    def base2():
        y, h = c_addnorm(F.linear(m_in, Wdn), res, wn); return F.linear(y, Wqkv), h
    def quack2():
        y, h = c_addnorm(gi.gemm(m_in, WdnT), res, wn); return gi.gemm(y, WqkvT), h
    def fold2():
        h = torch.empty(M, H, dtype=bf, device=dev)
        d, rstd = gi.gemm_rms(m_in, WdnT, C=res, norm_weight=wn, premult_out=h, eps=eps)
        return gi.gemm_norm_act(d, WqkvT, rstd=rstd, activation=None, store_preact=True)[0], h
    for f in (base2, quack2, fold2):
        try: f(); f()
        except Exception as e: print("warm fail", f.__name__, type(e).__name__, str(e).strip().splitlines()[-1][-200:], flush=True)
    step("c2_base", base2, chk2); step("c2_quack", quack2, chk2); step("c2_fold", fold2, chk2)
    step("p_cublas_dn", lambda: F.linear(m_in, Wdn)); y2, _ = c_addnorm(F.linear(m_in, Wdn), res, wn)
    step("p_cublas_qkv", lambda: F.linear(y2, Wqkv)); step("p_q_dn", lambda: gi.gemm(m_in, WdnT)); step("p_q_qkv", lambda: gi.gemm(y2, WqkvT))
    step("p_q_dn_rms", lambda: gi.gemm_rms(m_in, WdnT, C=res, norm_weight=wn, premult_out=hb, eps=eps))
    out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
