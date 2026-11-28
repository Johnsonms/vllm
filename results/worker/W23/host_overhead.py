"""Host-side cost of the prototype path: warm per-call wall time (no sync inside the loop) and the cost of a never-seen M
(prefill sees arbitrary M). Random weights, 8B shapes."""
import time, torch
from flashinfer.quantization.fp8_quantization import mxfp8_quantize
from quack import gemm_interface as gi
from quack.gemm_config import GemmConfig
from quack.epilogue.library import swiglu_quant_mod, identity_epi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
dev = "cuda"; e4, e8, bf = torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bfloat16
K, I, H = 4096, 12288, 4096
Bg = BlockScaledOperand.quantize(torch.randn(2 * I, K, device=dev) * 0.02, MXFP8_E4M3).mT; SFBg = Bg.scale.unsqueeze(0) if Bg.scale.ndim == 5 else Bg.scale
Bd = BlockScaledOperand.quantize(torch.randn(H, I, device=dev) * 0.02, MXFP8_E4M3).mT; SFBd = Bd.scale.unsqueeze(0) if Bd.scale.ndim == 5 else Bd.scale
def cfg(tm, tn, cm, dyn): return GemmConfig(tile_m=tm, tile_n=tn, cluster_m=cm, cluster_n=1, pingpong=False, is_dynamic_persistent=dyn)
C_SMALL, C_BIG = cfg(128, 256, 1, True), cfg(256, 256, 2, True)
def mlp(x):
    M = x.shape[0]; c = C_BIG if M >= 512 else C_SMALL; rm = -(-M // 128)
    q, sf = mxfp8_quantize(x, True); sf = sf.view(e8).view(1, rm, K // 128, 32, 4, 4)
    post = torch.empty(M, I, dtype=e4, device=dev); sfh = torch.empty(1, rm, I // 128, 32, 4, 4, dtype=e8, device=dev)
    swiglu_quant_mod(q, Bg.qdata, out={"postact": post}, store_d=False, tuned=False, config=c, postact_sf=sfh, SFA=sf, SFB=SFBg, bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
    y = torch.empty(M, H, dtype=bf, device=dev)
    identity_epi(post, Bd.qdata, out={"D": y}, tuned=False, config=c, SFA=sfh, SFB=SFBd, bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
    return y
def wall(fn, n=200):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    t1 = time.perf_counter(); torch.cuda.synchronize(); return (t1 - t0) / n * 1e6, (time.perf_counter() - t0) / n * 1e6
for M in (8, 512, 4096):
    x = torch.randn(M, K, device=dev, dtype=bf)
    t0 = time.perf_counter(); y = mlp(x); torch.cuda.synchronize(); first = time.perf_counter() - t0
    A = BlockScaledOperand.quantize(x, MXFP8_E4M3); ref = None
    h, tot = wall(lambda: mlp(x))
    print(f"M={M}: first call {first*1e3:.1f} ms; warm host time per MLP call {h:.1f} us (GPU-inclusive {tot:.1f} us)", flush=True)
# never-seen M values (prefill): first-call cost per new M after the configs are compiled
for M in (4097, 3001, 777, 4099, 1234):
    x = torch.randn(M, K, device=dev, dtype=bf); torch.cuda.synchronize(); t0 = time.perf_counter(); y = mlp(x); torch.cuda.synchronize()
    t1 = time.perf_counter(); y = mlp(x); torch.cuda.synchronize(); t2 = time.perf_counter()
    print(f"new M={M}: first {1e3*(t1-t0):.2f} ms, second {1e3*(t2-t1):.2f} ms, finite={bool(torch.isfinite(y.float()).all())}", flush=True)
