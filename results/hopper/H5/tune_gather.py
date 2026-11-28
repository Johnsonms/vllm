import os, sys, torch
os.environ["QUACK_PRINT_AUTOTUNING"] = "1"
from quack.gemm_interface import gemm_gated_out
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import kopt_quack_h5 as h5
torch.manual_seed(0); E, K, I, TOPK = 128, 2048, 768, 8
w1 = torch.randn(E, 2 * I, K, device="cuda", dtype=torch.bfloat16) / K**.5
for M in [int(m) for m in sys.argv[1:]]:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); lg = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)
    wts = torch.empty(M, TOPK, device="cuda"); ids = torch.empty(M, TOPK, device="cuda", dtype=torch.int32)
    ops.topk_softmax(wts, ids, torch.empty_like(ids), lg, True)
    cu, a_idx, inv, _ = h5.build_index(ids, E)
    h = torch.empty(M * TOPK, I, device="cuda", dtype=torch.bfloat16)
    print(f"=== M={M} FC1 gather", flush=True)
    gemm_gated_out(x, w1.transpose(1, 2), None, h, activation="swiglu", cu_seqlens_m=cu, A_idx=a_idx, concat_layout="B")
    torch.cuda.synchronize()
