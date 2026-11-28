"""H5: per-piece timing of a permute-free QuACK MoE vs current P2 and the SM90 default (CuTe-DSL)."""
import json, os, statistics as st, sys
os.environ.setdefault("KOPT_QUACK_MOE", "1")
import torch, flashinfer
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.core import ActivationType
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import kopt_quack as kq, kopt_quack_h5 as h5
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_permute, moe_unpermute
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import interleave_w13_up_gate_for_flashinfer_cutedsl_sm90
torch.manual_seed(0); dev = "cuda"; E, K, I, TOPK = 128, 2048, 768, 8
cfg = kq._configs()
from quack.gemm_config import GemmConfig
cfg_g = GemmConfig(tile_m=128, tile_n=256, pingpong=False, is_dynamic_persistent=False, cluster_m=2, cluster_n=1, swap_ab=False, device_capacity=9)
w1 = torch.randn(E, 2 * I, K, device=dev, dtype=torch.bfloat16) / K**.5
w2 = torch.randn(E, K, I, device=dev, dtype=torch.bfloat16) / I**.5
w1_fi = interleave_w13_up_gate_for_flashinfer_cutedsl_sm90(w1)
R = int(os.environ.get("ROUNDS", "5"))
def bench(fn, args):
    return st.median(st.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, input_args=args, repeat_time_ms=100)) for _ in range(R)) * 1e3
def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()
res = {}
for M in [int(m) for m in (sys.argv[2:] or [256, 1024, 4096, 8192])]:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); lg = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    wts = torch.empty(M, TOPK, device=dev); ids = torch.empty(M, TOPK, device=dev, dtype=torch.int32)
    ops.topk_softmax(wts, ids, torch.empty_like(ids), lg, True)
    rows = M * TOPK
    h = torch.empty(rows, I, device=dev, dtype=torch.bfloat16); y = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
    o = torch.empty(M, K, device=dev, dtype=torch.bfloat16); o2 = torch.empty_like(o); o_fi = torch.empty_like(o)
    px = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
    ws13 = torch.empty(rows * K, device=dev, dtype=torch.bfloat16); ws2 = torch.empty(rows * K, device=dev, dtype=torch.bfloat16)
    cu, a_idx, inv, wsrt = h5.build_index(ids, E, wts)
    # sanity on the index
    assert int(cu[-1]) == rows
    ff = lambda x, ids, wts, o: flashinfer.cute_dsl_fused_moe_bf16(x=x, token_selected_experts=ids, token_final_scales=wts, w1_weight=w1_fi, w2_weight=w2, num_experts=E, top_k=TOPK, num_local_experts=E, local_expert_offset=0, use_fused_finalize=False, moe_output=o, activation_type=int(ActivationType.Swiglu))
    with autotune(True): ff(x, ids, wts, o_fi)
    def new_total(x, ids, wts, o, h, y, use_w_in_fc2):
        cu, a_idx, inv, wsrt = h5.build_index(ids, E, wts if use_w_in_fc2 else None)
        h5.fc1_gather(x, w1, h, cu, a_idx, cfg["fc1"])
        if use_w_in_fc2:
            h5.fc2_w(h, w2, y, cu, wsrt, cfg["fc2"]); h5.combine(o, y, inv, TOPK)
        else:
            h5.fc2(h, w2, y, cu, cfg["fc2"]); h5.combine(o, y, inv, TOPK, wts)
    def new_total2(x, ids, wts, o, h, y, px):
        cu, a_idx, inv, _ = h5.build_index(ids, E)
        h5.gather_rows(x, a_idx, px)
        kq._configs(); __import__("quack.gemm_interface", fromlist=["x"])._gemm_act_call(px, w1.transpose(1, 2), None, h, None, None, activation="swiglu", cu_seqlens_m=cu, concat_layout=("B",), dynamic_scheduler=False, tuned=False, config=cfg["fc1"])
        h5.fc2(h, w2, y, cu, cfg["fc2"]); h5.combine(o, y, inv, TOPK, wts)
    def new_total3(x, ids, wts, o, h, y):
        cu, a_idx, inv, _ = h5.build_index(ids, E)
        h5.fc1_gather(x, w1, h, cu, a_idx, cfg_g)
        h5.fc2(h, w2, y, cu, cfg["fc2"]); h5.combine(o, y, inv, TOPK, wts)
    # correctness: new path (both variants) vs current P2 vs FI
    kq.quack_moe_experts(o2, x, w1, w2, wts, ids, E, ws13, ws2)
    new_total(x, ids, wts, o, h, y, False); e_a = rel(o, o2)
    new_total(x, ids, wts, o, h, y, True); e_b = rel(o, o2); e_fi = rel(o_fi, o2)
    new_total2(x, ids, wts, o, h, y, px); e_c = rel(o, o2); new_total3(x, ids, wts, o, h, y); e_d = rel(o, o2)
    print(f"relerr gather-copy {e_c:.4f}  A_idx-tuned {e_d:.4f}")
    pieces = {
        "default_cutedsl_total": bench(ff, (x, ids, wts, o_fi)),
        "p2_current_total": bench(lambda x, ids, wts, o, a, b: kq.quack_moe_experts(o, x, w1, w2, wts, ids, E, a, b), (x, ids, wts, o2, ws13, ws2)),
        "moe_permute(copy+index)": bench(lambda x, ids, px: moe_permute(x, None, ids, E, permuted_hidden_states=px), (x, ids, px)),
        "index_build_triton": bench(lambda ids, wts: h5.build_index(ids, E, wts), (ids, wts)),
        "fc1_contig(px)": bench(lambda px, h: kq._configs() and __import__("quack.gemm_interface", fromlist=["x"])._gemm_act_call(px, w1.transpose(1, 2), None, h, None, None, activation="swiglu", cu_seqlens_m=cu, concat_layout=("B",), dynamic_scheduler=False, tuned=False, config=cfg["fc1"]), (px, h)),
        "fc1_gather(A_idx)": bench(lambda x, h: h5.fc1_gather(x, w1, h, cu, a_idx, cfg["fc1"]), (x, h)),
        "fc1_gather(A_idx, tuned 128x256 coop)": bench(lambda x, h: h5.fc1_gather(x, w1, h, cu, a_idx, cfg_g), (x, h)),
        "gather_rows_triton(copy)": bench(lambda x, px: h5.gather_rows(x, a_idx, px), (x, px)),
        "new_total(idx+gather-copy+fc1+fc2+combine)": bench(lambda x, ids, wts, o, h, y, px: new_total2(x, ids, wts, o, h, y, px), (x, ids, wts, o, h, y, px)),
        "new_total(idx+A_idx tuned fc1+fc2+combine)": bench(lambda x, ids, wts, o, h, y: new_total3(x, ids, wts, o, h, y), (x, ids, wts, o, h, y)),
        "fc2": bench(lambda h, y: h5.fc2(h, w2, y, cu, cfg["fc2"]), (h, y)),
        "fc2_w(colvec)": bench(lambda h, y: h5.fc2_w(h, w2, y, cu, wsrt, cfg["fc2"]), (h, y)),
        "moe_unpermute": None,
        "combine_triton(w)": bench(lambda y, o: h5.combine(o, y, inv, TOPK, wts), (y, o)),
        "combine_triton(no w)": bench(lambda y, o: h5.combine(o, y, inv, TOPK), (y, o)),
        "new_total(w in combine)": bench(lambda x, ids, wts, o, h, y: new_total(x, ids, wts, o, h, y, False), (x, ids, wts, o, h, y)),
        "new_total(w in fc2)": bench(lambda x, ids, wts, o, h, y: new_total(x, ids, wts, o, h, y, True), (x, ids, wts, o, h, y)),
    }
    _, _, off, invp, _ = moe_permute(x, None, ids, E, permuted_hidden_states=px)
    pieces["moe_unpermute"] = bench(lambda y, o: moe_unpermute(o, y, wts, invp, off), (y, o))
    res[M] = dict(pieces, relerr_new_vs_p2=e_a, relerr_new_wfc2_vs_p2=e_b, relerr_default_vs_p2=e_fi)
    d = pieces["default_cutedsl_total"]
    print(f"--- M={M}  (relerr vs current P2: new {e_a:.4f}, new+w-in-fc2 {e_b:.4f}; default vs P2 {e_fi:.4f})")
    for k, v in pieces.items(): print(f"   {k:28s} {v:9.1f} us" + (f"   default/this = {d/v:.3f}x" if 'total' in k else ""))
json.dump(res, open(sys.argv[1], "w"), indent=1)
