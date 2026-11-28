"""Autotune QuACK FC1 (gated swiglu varlen) and FC2 (varlen gemm) on SM90 at Qwen3-30B-A3B shapes."""
import os, sys, json, torch
os.environ["QUACK_PRINT_AUTOTUNING"] = "1"
from quack.gemm_interface import gemm, gemm_gated_out
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute
torch.manual_seed(0)
E, K, I, TOPK = 128, 2048, 768, 8
w1 = torch.randn(E, 2 * I, K, device="cuda", dtype=torch.bfloat16) / K**0.5
w2 = torch.randn(E, K, I, device="cuda", dtype=torch.bfloat16) / I**0.5
for M in [int(m) for m in sys.argv[1:]]:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(M, E, device="cuda")
    ids = logits.topk(TOPK, -1).indices.to(torch.int32)
    px, _, off, inv, _ = moe_permute(x, None, ids, E)
    cu = off.to(torch.int32)
    h = torch.empty(M * TOPK, I, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(M * TOPK, K, device="cuda", dtype=torch.bfloat16)
    print(f"=== M={M} FC1", flush=True)
    gemm_gated_out(px, w1.transpose(1, 2), None, h, activation="swiglu", cu_seqlens_m=cu, concat_layout="B")
    print(f"=== M={M} FC2", flush=True)
    gemm(h, w2.transpose(1, 2), out=y, cu_seqlens_m=cu)
    torch.cuda.synchronize()
