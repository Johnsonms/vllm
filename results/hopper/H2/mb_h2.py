"""H2 kernel microbenches (cold L2, CUDA graph, CUPTI): P1 vs ops.topk_softmax; P2 vs
default FlashInfer CuTe-DSL SM90 experts and Triton fused_experts. Qwen3-30B-A3B shapes.
Run from the prototype tree with KOPT_QUACK_TOPK=1 KOPT_QUACK_MOE=1."""
import json, os, statistics as st, sys
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import kopt_quack as kq
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    interleave_w13_up_gate_for_flashinfer_cutedsl_sm90,
)

assert kq.P1_ON and kq.P2_ON
torch.manual_seed(0)
dev = "cuda"
E, K, I, TOPK = 128, 2048, 768, 8
ROUNDS = 7
out = {"p1": {}, "p2": {}}


def bench(fn, args):
    ts = []
    for _ in range(ROUNDS):
        t = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True,
                                      input_args=args, repeat_time_ms=100)
        ts.append(st.median(t))
    return st.median(ts) * 1e3  # us


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---------------- P1
for M in [1, 8, 64, 256, 1024, 4096, 8192]:
    g = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    w_ref = torch.empty(M, TOPK, device=dev, dtype=torch.float32)
    i_ref = torch.empty(M, TOPK, device=dev, dtype=torch.int32)
    tei = torch.empty(M, TOPK, device=dev, dtype=torch.int32)
    ops.topk_softmax(w_ref, i_ref, tei, g, True)
    w_q = torch.empty_like(w_ref); i_q = torch.empty_like(i_ref)
    kq.quack_topk_softmax(g, w_q, i_q)
    ids_eq = torch.equal(i_q, i_ref)
    if not ids_eq:  # tie order may differ; compare as sets
        ids_eq = torch.equal(i_q.sort(-1).values, i_ref.sort(-1).values)
        ids_eq = f"as-sets:{ids_eq}"
    werr = (w_q - w_ref).abs().max().item()

    def f_ref(g, w, i, t): ops.topk_softmax(w, i, t, g, True)
    def f_q(g, w, i): kq.quack_topk_softmax(g, w, i)
    tr = bench(f_ref, (g, w_ref, i_ref, tei)); tq = bench(f_q, (g, w_q, i_q))
    out["p1"][M] = dict(ref_us=tr, quack_us=tq, speedup=tr / tq, ids_identical=ids_eq, max_w_err=werr)
    print(f"P1 M={M}: topk_softmax {tr:.2f}us quack {tq:.2f}us {tr/tq:.2f}x ids={ids_eq} werr={werr:.1e}", flush=True)

# extra P1 alignment check: k=4/k=2 at M>1 (failed in stock 0.6.5)
for (EE, kk) in [(64, 4), (32, 2), (256, 8)]:
    g = torch.randn(37, EE, device=dev, dtype=torch.bfloat16)
    w = torch.empty(37, kk, device=dev); i = torch.empty(37, kk, device=dev, dtype=torch.int32)
    kq.quack_topk_softmax(g, w, i)
    wr = torch.empty_like(w); ir = torch.empty_like(i); ops.topk_softmax(wr, ir, torch.empty_like(i), g, True)
    print(f"P1 E={EE} k={kk} M=37: ids_eq={torch.equal(i, ir)} werr={(w-wr).abs().max().item():.1e}")

# ---------------- P2
w1 = torch.randn(E, 2 * I, K, device=dev, dtype=torch.bfloat16) / K**0.5
w2 = torch.randn(E, K, I, device=dev, dtype=torch.bfloat16) / I**0.5
w1_fi = interleave_w13_up_gate_for_flashinfer_cutedsl_sm90(w1)
import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.core import ActivationType
act = int(ActivationType.Swiglu)


def ref_moe(x, ids, wts):
    o = torch.zeros(x.size(0), K, device=dev, dtype=torch.float32)
    xf = x.float()
    for e in range(E):
        t, s = (ids == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        y = xf[t] @ w1[e].float().t()
        h = torch.nn.functional.silu(y[:, :I]) * y[:, I:]
        o.index_add_(0, t, (h @ w2[e].float().t()) * wts[t, s, None])
    return o


for M in [64, 256, 1024, 2048, 4096, 8192]:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    wts = torch.empty(M, TOPK, device=dev); ids = torch.empty(M, TOPK, device=dev, dtype=torch.int32)
    ops.topk_softmax(wts, ids, torch.empty_like(ids), logits, True)
    ws13 = torch.empty(M * TOPK * max(I, K), device=dev, dtype=torch.bfloat16)
    ws2 = torch.empty(M * TOPK * max(2 * I, K), device=dev, dtype=torch.bfloat16)
    o_q = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    o_fi = torch.empty_like(o_q)

    def f_q(x, ids, wts, o, a, b):
        kq.quack_moe_experts(o, x, w1, w2, wts, ids, E, a, b)

    def f_tr(x, ids, wts):
        return fused_experts(x, w1, w2, wts, ids)

    def f_fi(x, ids, wts, o):
        flashinfer.cute_dsl_fused_moe_bf16(x=x, token_selected_experts=ids, token_final_scales=wts,
                                           w1_weight=w1_fi, w2_weight=w2, num_experts=E, top_k=TOPK,
                                           num_local_experts=E, local_expert_offset=0,
                                           use_fused_finalize=False, moe_output=o, activation_type=act)
    with autotune(True):
        f_fi(x, ids, wts, o_fi)
    torch.cuda.synchronize()
    f_q(x, ids, wts, o_q, ws13, ws2); f_fi(x, ids, wts, o_fi); o_tr = f_tr(x, ids, wts)
    ref = ref_moe(x, ids, wts)
    e_q, e_fi, e_tr = rel(o_q, ref), rel(o_fi, ref), rel(o_tr, ref)
    t_q = bench(f_q, (x, ids, wts, o_q, ws13, ws2))
    t_fi = bench(f_fi, (x, ids, wts, o_fi))
    t_tr = bench(f_tr, (x, ids, wts))
    out["p2"][M] = dict(quack_us=t_q, cutedsl_default_us=t_fi, triton_us=t_tr,
                        vs_default=t_fi / t_q, vs_triton=t_tr / t_q,
                        relerr_quack=e_q, relerr_cutedsl=e_fi, relerr_triton=e_tr)
    print(f"P2 M={M}: quack {t_q:.1f}us  cutedsl(default) {t_fi:.1f}us ({t_fi/t_q:.2f}x)  triton {t_tr:.1f}us ({t_tr/t_q:.2f}x)"
          f"  relerr q/fi/tr {e_q:.4f}/{e_fi:.4f}/{e_tr:.4f}", flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
