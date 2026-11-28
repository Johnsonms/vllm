"""gemm -> mxfp8 quantized output: wrong/nonfinite results. Does it depend on tuned config, input kind, M/N? 5 reps each."""
import torch
from quack import gemm_interface as gi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
dev = "cuda"
def run(M, K, N, inp, tuned):
    torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev) * 0.02
    ref = x.float() @ w.T
    if inp == "mxfp8": A, B = BlockScaledOperand.quantize(x, MXFP8_E4M3), BlockScaledOperand.quantize(w, MXFP8_E4M3).mT
    else: A, B = x, w.bfloat16().T.contiguous()
    rels = []
    for rep in range(5):
        try:
            d = gi.gemm(A, B, out_dtype=MXFP8_E4M3, tuned=tuned).dequantize().float(); rels.append(round(((d - ref).norm() / ref.norm()).item(), 4))
        except Exception as e: rels.append(type(e).__name__ + ":" + str(e).strip().splitlines()[-1][:80]); break
    print(f"M={M} K={K} N={N} in={inp} tuned={tuned}: {rels}", flush=True)
for tuned in (True, False):
    for inp in ("mxfp8", "bf16"):
        for (M, K, N) in [(8, 4096, 6144), (8, 4096, 24576), (512, 4096, 6144), (512, 4096, 4096), (128, 4096, 4096), (2048, 4096, 24576), (512, 12288, 4096)]:
            run(M, K, N, inp, tuned)
