import torch, quack
from quack import gemm_interface as gi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
torch.manual_seed(0); dev = "cuda"
M, K, I = 32, 4096, 12288
x = torch.randn(M, K, device=dev); w = torch.randn(2 * I, K, device=dev) * 0.02   # rows [gate; up]
pre = x @ w.T; gref = torch.nn.functional.silu(pre[:, :I]) * pre[:, I:]
def grel(a): return ((a.float() - gref).norm() / gref.norm()).item()
def case(name, fn):
    try: print(f"OK   {name}: {fn()}", flush=True)
    except Exception as e: print(f"FAIL {name}: {type(e).__name__}: {str(e).strip().splitlines()[-1][:300]}", flush=True)
xb = x.bfloat16(); wbT = w.bfloat16().T   # (K, 2I) view of the (2I, K) weight, rows [gate; up]
wi = gi._concat_interleave(w)   # (2I, K) rows g0,u0,g1,u1,...
A8 = BlockScaledOperand.quantize(x, MXFP8_E4M3); B8 = BlockScaledOperand.quantize(wi, MXFP8_E4M3).mT
CL = ("B",)
def g(A, B, **kw):
    cl = CL if isinstance(B, torch.Tensor) else None
    _, post = gi.gemm_act(A, B, activation="swiglu", store_preact=False, concat_layout=cl, **kw); return post
case("gated bf16 -> bf16", lambda: grel(g(xb, wbT)))
case("gated mxfp8 x mxfp8 -> bf16", lambda: grel(g(A8, B8)))
case("gated mxfp8 x mxfp8 -> mxfp8 postact (quant out)", lambda: g(A8, B8, postact_dtype=MXFP8_E4M3))
case("gated bf16 -> mxfp8 postact", lambda: g(xb, wbT, postact_dtype=MXFP8_E4M3))
case("gated mxfp8 -> e4m3 postact (plain dtype, no scales)", lambda: tuple(g(A8, B8, postact_dtype=torch.float8_e4m3fn).shape))
