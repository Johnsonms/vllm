"""W21 addendum: gate_up chain INCLUDING the quant of h. vLLM = DeepGEMM + Inductor silu-mul + packed 1x128 quant kernel (as serving).
QuACK = swiglu_quant_mod (gated GEMM, MXFP8 postact + scales written by the epilogue), config pinned by hand because the
autotuned configs give wrong results. usage: bench2.py SNAPSHOT LAYER OUT.jsonl"""
import json, sys, torch, torch.nn.functional as F
from safetensors import safe_open
from flashinfer.testing import bench_gpu_time_with_cupti
import vllm._custom_ops  # noqa
import vllm.envs as envs
from vllm.model_executor.kernels.linear.scaled_mm import deep_gemm as _dgk  # noqa
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
from vllm.config import VllmConfig, set_current_vllm_config
from quack import gemm_interface as gi
from quack.gemm_config import GemmConfig
from quack.epilogue.library import swiglu_quant_mod
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
snap, layer, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = "cuda"; E8 = is_deep_gemm_e8m0_used(); e4, e8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{snap}/{idx[name]}", "pt", device=dev) as f: return f.get_tensor(name)
p = f"model.layers.{layer}.mlp."
wq = torch.cat([load(p + n + ".weight") for n in ("gate_proj", "up_proj")]); ws = torch.cat([load(p + n + ".weight_scale_inv") for n in ("gate_proj", "up_proj")]).float()
N, K = wq.shape; I = N // 2
wdeq = wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
W, Ws = deepgemm_post_process_fp8_weight_block(wq.clone(), ws.clone(), (128, 128), E8)
B8 = BlockScaledOperand.quantize(gi._concat_interleave(wdeq), MXFP8_E4M3).mT
SFB = B8.scale if B8.scale.ndim == 6 else B8.scale.unsqueeze(0)
with set_current_vllm_config(VllmConfig()):
    quant = QuantFP8(static=False, group_shape=GroupShape(1, 128), use_ue8m0=E8, tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES, column_major_scales=True)
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8_packed_for_deepgemm as _pq
quant = lambda a: _pq(a, 128, use_ue8m0=True)   # W24 fix: the packed kernel serving uses
def t(fn): return round(1e3 * float(torch.tensor(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=False, repeat_time_ms=60)).median()), 2)
CANDS = [GemmConfig(tile_m=tm, tile_n=tn, cluster_m=cm, cluster_n=1, pingpong=False, is_dynamic_persistent=dyn)
         for (tm, cm) in ((128, 1), (256, 2)) for tn in (64, 128, 192, 256) for dyn in (False, True)]
out = open(outp, "a")
for M in (1, 8, 32, 128, 512, 2048, 8192):
    torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    pre = x.float() @ wdeq.T; gref = F.silu(pre[:, :I]) * pre[:, I:]; rel = lambda a: round(((a.float() - gref).norm() / gref.norm()).item(), 4)
    xq, xs = quant(x)
    torch._dynamo.reset(); silu = torch.compile(lambda o: F.silu(o[:, :I]) * o[:, I:], dynamic=False)
    def v_chain():
        o = torch.empty((M, N), dtype=torch.bfloat16, device=dev); torch.ops.vllm.fp8_gemm_nt_op(xq, xs, W, Ws, o, E8)
        return quant(silu(o))
    v_chain(); v_chain()
    hq, hs = v_chain(); h_bf16 = silu((lambda o: (torch.ops.vllm.fp8_gemm_nt_op(xq, xs, W, Ws, o, E8), o)[1])(torch.empty((M, N), dtype=torch.bfloat16, device=dev)))
    rec = dict(N=N, K=K, M=M, v_chain_us=t(v_chain), v_h_err_before_quant=rel(h_bf16))
    A8 = BlockScaledOperand.quantize(x, MXFP8_E4M3); SFA = A8.scale if A8.scale.ndim == 6 else A8.scale.unsqueeze(0)
    post = torch.empty(M, I, dtype=e4, device=dev); sf = torch.empty(1, -(-M // 128), -(-I // 128), 32, 4, 4, dtype=e8, device=dev)
    res = []
    for c in CANDS:
        def q():
            swiglu_quant_mod(A8.qdata, B8.qdata, out={"postact": post}, store_d=False, tuned=False, config=c, postact_sf=sf,
                             SFA=SFA, SFB=SFB, bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
        tag = f"{c.tile_m}x{c.tile_n} c{c.cluster_m} dyn{int(c.is_dynamic_persistent)}"
        try:
            errs = []
            for _ in range(3):
                q(); errs.append(rel(BlockScaledOperand.from_parts(post, sf[0], MXFP8_E4M3).dequantize()))
            ok = all(e == e and e < 0.08 for e in errs)
            res.append(dict(cfg=tag, ok=ok, err=errs[0] if ok else errs, us=t(q)))
        except Exception as e:
            res.append(dict(cfg=tag, ok=False, fail=f"{type(e).__name__}: {str(e).strip().splitlines()[-1][-120:]}"))
    good = [r for r in res if r["ok"]]; best = min(good, key=lambda r: r["us"]) if good else None
    rec.update(q_best=best, q_all=res, speedup=round(rec["v_chain_us"] / best["us"], 3) if best else None)
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "q_all"}), "| wrong:", [r["cfg"] for r in res if not r["ok"] and "fail" not in r], "| failed:", sorted({r["fail"][:60] for r in res if "fail" in r}), flush=True)
