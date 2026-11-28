"""W21 step 2: which FP8 call shapes QuACK 0.6.5 accepts on SM100. Each case prints OK (+ rel. error vs fp32 ref) or the error."""
import torch, traceback, quack
from quack import gemm_interface as gi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
torch.manual_seed(0); dev = "cuda"
M, K, N = 32, 4096, 8192
x = torch.randn(M, K, device=dev); w = torch.randn(N, K, device=dev) * 0.02
ref = x @ w.T
def rel(a): return ((a.float() - ref).norm() / ref.norm()).item()
def case(name, fn):
    try:
        r = fn(); print(f"OK   {name}: {r}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)

xb, wb = x.bfloat16(), w.bfloat16()
A8 = BlockScaledOperand.quantize(x, MXFP8_E4M3)            # (M,K) scales along K
B8 = BlockScaledOperand.quantize(w, MXFP8_E4M3).mT        # (K,N) view, scales along K
case("bf16 x bf16 -> bf16", lambda: rel(gi.gemm(xb, wb.T.contiguous())))
case("plain e4m3 x e4m3 (no scales) -> bf16", lambda: tuple(gi.gemm(x.clamp(-4,4).to(torch.float8_e4m3fn), (w*8).to(torch.float8_e4m3fn).T.contiguous(), out_dtype=torch.bfloat16).shape))
case("mxfp8 x mxfp8 -> bf16", lambda: rel(gi.gemm(A8, B8)))
case("bf16 A x mxfp8 B -> bf16 (weight-only)", lambda: rel(gi.gemm(xb, B8)))
def qout():
    o = gi.gemm(A8, B8, out_dtype=MXFP8_E4M3); return f"{type(o).__name__} rel={rel(o.dequantize())}"
case("mxfp8 x mxfp8 -> mxfp8 out (quant_out)", qout)
def qout_bf16in():
    o = gi.gemm(xb, wb.T.contiguous(), out_dtype=MXFP8_E4M3); return f"rel={rel(o.dequantize())}"
case("bf16 x bf16 -> mxfp8 out", qout_bf16in)
# gated: weight (2I, K) gate/up
I = N // 2
gref = torch.nn.functional.silu(ref[:, :I]) * ref[:, I:]
def grel(a): return ((a.float() - gref).norm() / gref.norm()).item()
import inspect
print("gemm_act activations:", sorted(k for k in list(gi.gated_to_pytorch_fn_map) ), flush=True)
print("gemm_gated_out sig:", inspect.signature(gi.gemm_gated_out), flush=True)
print("mlp/linear API:", [n for n in dir(quack) if not n.startswith('_')][:80], flush=True)
