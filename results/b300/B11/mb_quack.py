# B11 task 3 microbench on one B300 (SM103): QuACK vs the kernels vLLM runs today.
# Timing: CUDA graph of REP calls, L2 flushed (256 MB write) before each replay, median of 7
# interleaved rounds x 5 replays; reported per call in us.
import json, statistics as st, sys, time
import torch

torch.manual_seed(0)
dev = "cuda"
REP = 20
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REP):
            fn()
    return g


def time_graphs(graphs, rounds=7, reps=5):
    res = {k: [] for k in graphs}
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(rounds):
        for k, g in graphs.items():
            for _ in range(reps):
                flush.zero_()
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                res[k].append(e0.elapsed_time(e1) * 1000 / REP)
    return {k: st.median(v) for k, v in res.items()}


def jit_time(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return time.perf_counter() - t


out = {"norm": [], "softmax": [], "sampler": []}

# ---------------- RMSNorm / fused add + RMSNorm ----------------
from vllm import _custom_ops as ops
import flashinfer
from quack.rmsnorm import rmsnorm_fwd

Ms = [1, 2, 4, 8, 16, 256, 1024, 4096, 8192]
for H in (2048, 4096, 5120):
    w = (torch.randn(H, device=dev) * 0.1).to(torch.bfloat16)
    for M in Ms:
        x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        r = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        row = {"H": H, "M": M}
        # Reference in fp32 (Gemma style: x_hat * (1 + w)) and plain (x_hat * w).
        def ref(xx, rr, gemma):
            h = (xx.float() + rr.float()) if rr is not None else xx.float()
            y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
            return (y * ((1 + w.float()) if gemma else w.float())), h
        # --- plain RMSNorm ---
        o_v = torch.empty_like(x)
        f_v = lambda: ops.rms_norm(o_v, x, w, 1e-6)
        f_fi = lambda: flashinfer.rmsnorm(x, w, 1e-6)
        if M == Ms[0]:
            row["quack_jit_s_rms"] = jit_time(lambda: rmsnorm_fwd(x, w, eps=1e-6))
        f_q = lambda: rmsnorm_fwd(x, w, eps=1e-6)
        yr, _ = ref(x, None, False)
        f_v(); yq = f_q()[0]; yfi = f_fi()
        row["rms_maxdiff_vs_ref"] = {"vllm": (o_v.float() - yr).abs().max().item(),
                                     "flashinfer": (yfi.float() - yr).abs().max().item(),
                                     "quack": (yq.float() - yr).abs().max().item()}
        row["rms_quack_eq_vllm"] = torch.equal(yq, o_v)
        # --- fused add + Gemma RMSNorm (what Qwen3.5/3.6 runs: flashinfer gemma_fused_add_rmsnorm) ---
        xa, ra = x.clone(), r.clone()
        f_fa_fi = lambda: flashinfer.gemma_fused_add_rmsnorm(xa, ra, w, 1e-6)
        xb, rb = x.clone(), r.clone()
        f_fa_v = lambda: ops.fused_add_rms_norm(xb, rb, w, 1e-6)  # plain-weight variant, for reference
        if M == Ms[0]:
            row["quack_jit_s_add"] = jit_time(lambda: rmsnorm_fwd(x, w, residual=r, eps=1e-6, weight_offset=1.0))
        f_fa_q = lambda: rmsnorm_fwd(x, w, residual=r, eps=1e-6, weight_offset=1.0)
        yg, hg = ref(x, r, True)
        xa.copy_(x); ra.copy_(r); flashinfer.gemma_fused_add_rmsnorm(xa, ra, w, 1e-6)
        yq2, rq2, _ = f_fa_q()
        row["add_maxdiff_vs_ref"] = {"flashinfer_gemma": (xa.float() - yg).abs().max().item(),
                                     "quack": (yq2.float() - yg).abs().max().item()}
        row["add_quack_eq_flashinfer"] = torch.equal(yq2, xa)
        row["add_residual_eq"] = torch.equal(rq2, ra)
        try:
            gs = {"rms_vllm": graph_of(f_v), "rms_flashinfer": graph_of(f_fi), "rms_quack": graph_of(f_q),
                  "add_flashinfer_gemma": graph_of(f_fa_fi), "add_vllm_plain": graph_of(f_fa_v),
                  "add_quack_gemma": graph_of(f_fa_q)}
            row["graph_ok"] = True
            row["us"] = time_graphs(gs)
        except Exception as e:  # noqa
            row["graph_ok"] = False; row["err"] = repr(e)[:300]
        out["norm"].append(row)
        print(json.dumps(row), flush=True)

# ---------------- softmax over the vocab, and the sampler ----------------
from quack.softmax import softmax_fwd
from vllm.v1.sample.ops.topk_topp_sampler import flashinfer_sample
V = 248320
for M in (1, 8, 32):
    lg = torch.randn(M, V, device=dev, dtype=torch.float32) * 3
    row = {"M": M, "V": V}
    row["quack_jit_s"] = jit_time(lambda: softmax_fwd(lg))
    pt, pq = torch.softmax(lg, -1), softmax_fwd(lg)
    row["softmax_maxdiff"] = (pt - pq).abs().max().item()
    k = torch.full((M,), 20, dtype=torch.int32, device=dev)
    p = torch.full((M,), 0.95, dtype=torch.float32, device=dev)
    f_cur = lambda: flashinfer_sample(lg, k, p)
    f_fast = lambda: flashinfer.sampling.top_k_top_p_sampling_from_logits(lg, k, p, deterministic=True, top_k_max=20)
    # Distribution check on row 0: both draws must come from row 0's top-20.
    top20 = set(torch.topk(lg[0], 20).indices.tolist())
    row["fast_in_top20"] = all(int(flashinfer.sampling.top_k_top_p_sampling_from_logits(
        lg, k, p, deterministic=True, top_k_max=20)[0]) in top20 for _ in range(200))
    gs = {"torch_softmax": graph_of(lambda: torch.softmax(lg, -1)), "quack_softmax": graph_of(lambda: softmax_fwd(lg))}
    try:
        gs["sampler_current"] = graph_of(f_cur); gs["sampler_topk_max"] = graph_of(f_fast); row["graph_ok"] = True
    except Exception as e:  # noqa
        row["graph_ok"] = False; row["err"] = repr(e)[:300]
    row["us"] = time_graphs(gs)
    out["softmax"].append(row)
    print(json.dumps(row), flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1)
