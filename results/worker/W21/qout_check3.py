import torch
from quack import gemm_interface as gi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
dev = "cuda"
for (M, K, N, inp) in [(512, 4096, 6144, "mxfp8"), (512, 4096, 6144, "bf16"), (512, 4096, 4096, "mxfp8")]:
    torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev) * 0.02
    ref = x.float() @ w.T
    A, B = (BlockScaledOperand.quantize(x, MXFP8_E4M3), BlockScaledOperand.quantize(w, MXFP8_E4M3).mT) if inp == "mxfp8" else (x, w.bfloat16().T.contiguous())
    print(f"CASE M={M} N={N} in={inp}", flush=True)
    d = gi.gemm(A, B, out_dtype=MXFP8_E4M3).dequantize().float()
    print("RESULT rel", ((d - ref).norm() / ref.norm()).item(), flush=True)
