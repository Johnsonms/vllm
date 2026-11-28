"""W21 correction: the epilogue library has swiglu_quant_mod (gated + quantized postact). Does it run and is it right?"""
import torch, torch.nn.functional as F, traceback
from quack import gemm_interface as gi
from quack.epilogue.library import swiglu_quant_mod, swiglu_mod
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
dev = "cuda"; e4, e8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
def run(M, K, I, inp, tuned):
    torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(2 * I, K, device=dev) * 0.02
    pre = x.float() @ w.T; gref = F.silu(pre[:, :I]) * pre[:, I:]
    wi = gi._concat_interleave(w)
    post = torch.empty(M, I, dtype=e4, device=dev); sf = torch.empty(1, -(-M // 128), -(-I // 128), 32, 4, 4, dtype=e8, device=dev)
    kw = {}
    if inp == "mxfp8":
        A8 = BlockScaledOperand.quantize(x, MXFP8_E4M3); B8 = BlockScaledOperand.quantize(wi, MXFP8_E4M3).mT
        A, B = A8.qdata, B8.qdata; kw = dict(SFA=A8.scale if A8.scale.ndim == 6 else A8.scale.unsqueeze(0), SFB=B8.scale if B8.scale.ndim == 6 else B8.scale.unsqueeze(0), bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
    else: A, B = x, wi.bfloat16().T
    rels = []
    for rep in range(4):
        try:
            swiglu_quant_mod(A, B, out={"postact": post}, store_d=False, tuned=tuned, postact_sf=sf, **kw)
            d = BlockScaledOperand.from_parts(post, sf[0], MXFP8_E4M3).dequantize().float()
            rels.append(round(((d - gref).norm() / gref.norm()).item(), 4))
        except Exception as e:
            rels.append(f"{type(e).__name__}: {str(e).strip().splitlines()[-1][-260:]}"); break
    print(f"M={M} K={K} I={I} in={inp} tuned={tuned}: {rels}", flush=True)
for tuned in (False, True):
    for inp in ("mxfp8", "bf16"):
        for (M, K, I) in [(8, 4096, 12288), (128, 4096, 12288), (512, 4096, 12288), (2048, 4096, 12288), (512, 5120, 25600)]:
            run(M, K, I, inp, tuned)
