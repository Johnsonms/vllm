"""W23: serving-like kernel bench. One CUDA graph runs the MLP of L consecutive layers (distinct real weights), so weights
are never L2-resident and nothing is artificially flushed. us per layer. vLLM path vs prototype with a config sweep.
usage: multilayer.py SNAPSHOT OUT.jsonl [Ms]"""
import json, sys, torch, torch.nn.functional as F
from safetensors import safe_open
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.quantization.fp8_quantization import mxfp8_quantize
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
from quack.epilogue.library import swiglu_quant_mod, identity_epi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
snap, outp = sys.argv[1], sys.argv[2]; Ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 16, 32, 64, 128, 256, 512, 1024]
import os
LAYERS = list(range(8, 8 + int(os.environ.get("NL", "8")))); L = len(LAYERS)
dev = "cuda"; E8 = is_deep_gemm_e8m0_used(); e4, e8, bf = torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bfloat16
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{snap}/{idx[name]}", "pt", device=dev) as f: return f.get_tensor(name)
def wpair(layer, names):
    p = f"model.layers.{layer}.mlp."
    wq = torch.cat([load(p + n + ".weight") for n in names]); ws = torch.cat([load(p + n + ".weight_scale_inv") for n in names]).float()
    return wq, ws, wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)[:wq.shape[0], :wq.shape[1]]
six = lambda s: s if s.ndim == 6 else s.unsqueeze(0)
V, P = [], []
for l in LAYERS:
    gq, gs, gdeq = wpair(l, ["gate_proj", "up_proj"]); dq, ds, ddeq = wpair(l, ["down_proj"])
    V.append(deepgemm_post_process_fp8_weight_block(gq, gs, (128, 128), E8) + deepgemm_post_process_fp8_weight_block(dq, ds, (128, 128), E8))
    bg = BlockScaledOperand.quantize(gi._concat_interleave(gdeq), MXFP8_E4M3).mT; bd = BlockScaledOperand.quantize(ddeq, MXFP8_E4M3).mT
    P.append((bg.qdata, six(bg.scale), bd.qdata, six(bd.scale))); del gdeq, ddeq
N, K = V[0][0].shape; I = N // 2; H = V[0][2].shape[0]
with set_current_vllm_config(VllmConfig()):
    quant = QuantFP8(static=False, group_shape=GroupShape(1, 128), use_ue8m0=E8, tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES, column_major_scales=True)
def t(fn): return round(1e3 * float(torch.tensor(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=False, repeat_time_ms=100)).median()) / L, 2)
def cfg(tm, tn, cm, dyn): return GemmConfig(tile_m=tm, tile_n=tn, cluster_m=cm, cluster_n=1, pingpong=False, is_dynamic_persistent=dyn)
CANDS = {f"{tm}x{tn}c{cm}d{int(d)}": cfg(tm, tn, cm, d) for (tm, cm) in ((128, 1), (256, 2)) for tn in (64, 128, 256) for d in (False, True)}
fmt = dict(bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
out = open(outp, "a")
for M in Ms:
    torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=bf); rm = -(-M // 128)
    torch._dynamo.reset(); silu = torch.compile(lambda o: F.silu(o[:, :I]) * o[:, I:], dynamic=False)
    def v_all():
        y = x
        for (Wg, Wgs, Wd, Wds) in V:
            q, s = quant(y); o = torch.empty((M, N), dtype=bf, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q, s, Wg, Wgs, o, E8)
            q2, s2 = quant(silu(o)); y = torch.empty((M, H), dtype=bf, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q2, s2, Wd, Wds, y, E8)
        return y
    def p_all(cg, cd):
        def f():
            y = x
            for (bg, sfbg, bd, sfbd) in P:
                q, sf = mxfp8_quantize(y, True); sf = sf.view(e8).view(1, rm, K // 128, 32, 4, 4)
                post = torch.empty(M, I, dtype=e4, device=dev); sfh = torch.empty(1, rm, I // 128, 32, 4, 4, dtype=e8, device=dev)
                swiglu_quant_mod(q, bg, out={"postact": post}, store_d=False, tuned=False, config=cg, postact_sf=sfh, SFA=sf, SFB=sfbg, **fmt)
                y = torch.empty(M, H, dtype=bf, device=dev); identity_epi(post, bd, out={"D": y}, tuned=False, config=cd, SFA=sfh, SFB=sfbd, **fmt)
            return y
        return f
    print("v_all 1", flush=True); v_all(); torch.cuda.synchronize(); print("v_all ok", flush=True); v_all(); rec = dict(M=M, v_us=t(v_all), g={}, d={}); print("v timed", rec, flush=True)
    base_c = CANDS["128x256c1d1"]
    for name, c in CANDS.items():          # sweep the gated config with down fixed, then the down config with gated fixed
        print("try g", M, name, flush=True)
        try: f = p_all(c, base_c); f(); torch.cuda.synchronize(); rec["g"][name] = t(f)
        except Exception as e: rec["g"][name] = None
    bg_name = min((k for k, v in rec["g"].items() if v), key=lambda k: rec["g"][k])
    for name, c in CANDS.items():
        print("try d", M, name, flush=True)
        try: f = p_all(CANDS[bg_name], c); f(); torch.cuda.synchronize(); rec["d"][name] = t(f)
        except Exception as e: rec["d"][name] = None
    bd_name = min((k for k, v in rec["d"].items() if v), key=lambda k: rec["d"][k])
    yv = v_all().float(); yp = p_all(CANDS[bg_name], CANDS[bd_name])().float()
    rec.update(best_g=bg_name, best_d=bd_name, p_us=rec["d"][bd_name], speedup=round(rec["v_us"] / rec["d"][bd_name], 3), rel_diff_after_8_layers=round(((yp - yv).norm() / yv.norm()).item(), 4))
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(json.dumps({k: v for k, v in rec.items() if k not in ("g", "d")}), "| pinned-in-prototype 128x256c1d1:", rec["d"].get("128x256c1d1") if bg_name == "128x256c1d1" else rec["g"].get("128x256c1d1"), flush=True)
