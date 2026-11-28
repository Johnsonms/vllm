"""W23 step 0 (kernels only): whole MLP, bf16 x in -> bf16 out, real Qwen3-8B-FP8 layer weights.
vLLM  : packed quant -> DeepGEMM gate_up -> Inductor silu*mul -> packed quant -> DeepGEMM down        (5 kernels)
proto : FlashInfer mxfp8_quantize (128x4 swizzled scales) -> QuACK swiglu_quant_mod -> QuACK MXFP8 GEMM down   (3 kernels)
usage: mlp_kernel.py SNAPSHOT LAYER OUT.jsonl"""
import json, sys, torch, torch.nn.functional as F
from safetensors import safe_open
from flashinfer.testing import bench_gpu_time_with_cupti
import flashinfer
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
from quack.epilogue.library import swiglu_quant_mod
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
snap, layer, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = "cuda"; E8 = is_deep_gemm_e8m0_used(); e4, e8, bf = torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bfloat16
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{snap}/{idx[name]}", "pt", device=dev) as f: return f.get_tensor(name)
def wpair(names):
    p = f"model.layers.{layer}.mlp."
    wq = torch.cat([load(p + n + ".weight") for n in names]); ws = torch.cat([load(p + n + ".weight_scale_inv") for n in names]).float()
    return wq, ws, wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)[:wq.shape[0], :wq.shape[1]]
gq, gs, gdeq = wpair(["gate_proj", "up_proj"]); dq, ds, ddeq = wpair(["down_proj"]); N, K = gq.shape; I = N // 2; H = dq.shape[0]
Wg, Wgs = deepgemm_post_process_fp8_weight_block(gq.clone(), gs.clone(), (128, 128), E8)
Wd, Wds = deepgemm_post_process_fp8_weight_block(dq.clone(), ds.clone(), (128, 128), E8)
Bg = BlockScaledOperand.quantize(gi._concat_interleave(gdeq), MXFP8_E4M3).mT; SFBg = Bg.scale if Bg.scale.ndim == 6 else Bg.scale.unsqueeze(0)
Bd = BlockScaledOperand.quantize(ddeq, MXFP8_E4M3).mT
with set_current_vllm_config(VllmConfig()):
    quant = QuantFP8(static=False, group_shape=GroupShape(1, 128), use_ue8m0=E8, tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES, column_major_scales=True)
def t(fn): return round(1e3 * float(torch.tensor(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=60)).median()), 2)
def cfg(tm, tn, cm, dyn): return GemmConfig(tile_m=tm, tile_n=tn, cluster_m=cm, cluster_n=1, pingpong=False, is_dynamic_persistent=dyn)
GCANDS = [cfg(128, 256, 1, False), cfg(128, 256, 1, True), cfg(128, 128, 1, False), cfg(128, 128, 1, True), cfg(256, 256, 2, False), cfg(256, 256, 2, True)]
out = open(outp, "a")
for M in (1, 8, 32, 128, 512, 2048, 4096, 8192):
    torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=bf)
    pre = x.float() @ gdeq.T; ref = (F.silu(pre[:, :I]) * pre[:, I:]) @ ddeq.T; rel = lambda a: round(((a.float() - ref).norm() / ref.norm()).item(), 4)
    torch._dynamo.reset(); silu = torch.compile(lambda o: F.silu(o[:, :I]) * o[:, I:], dynamic=False)
    def v_mlp():
        q, s = quant(x); o = torch.empty((M, N), dtype=bf, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q, s, Wg, Wgs, o, E8)
        q2, s2 = quant(silu(o)); y = torch.empty((M, H), dtype=bf, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q2, s2, Wd, Wds, y, E8); return y
    v_mlp(); v_mlp()
    rec = dict(M=M, v_err=rel(v_mlp()), v_us=t(v_mlp))
    rm, rkx, rkh = -(-M // 128), K // 128, I // 128
    def fi_quant():
        q, sf = mxfp8_quantize(x, True); return q, sf.view(e8).view(1, rm, rkx, 32, 4, 4)
    try:
        q0, sf0 = fi_quant(); A_ref = BlockScaledOperand.quantize(x, MXFP8_E4M3)
        rec["fi_quant_matches_quack"] = bool((q0.view(torch.uint8) == A_ref.qdata.view(torch.uint8)).all()) and bool((sf0.view(torch.uint8).flatten()[: A_ref.scale.numel()] == A_ref.scale.view(torch.uint8).flatten()).all())
        rec["fi_quant_us"] = t(lambda: fi_quant()[0]); rec["v_quant_us"] = t(lambda: quant(x)[0])
    except Exception as e: rec["fi_quant_fail"] = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][-200:]}"
    best = None
    for c in GCANDS:
        post = torch.empty(M, I, dtype=e4, device=dev); sfh = torch.empty(1, rm, rkh, 32, 4, 4, dtype=e8, device=dev)
        def p_mlp():
            q, sf = fi_quant()
            swiglu_quant_mod(q, Bg.qdata, out={"postact": post}, store_d=False, tuned=False, config=c, postact_sf=sfh, SFA=sf, SFB=SFBg, bs_format_a="mxfp8_e4m3", bs_format_b="mxfp8_e4m3")
            return gi.gemm(BlockScaledOperand.from_parts(post, sfh[0], MXFP8_E4M3), Bd)
        try:
            p_mlp(); err = rel(p_mlp()); us = t(p_mlp)
            if err < 0.12 and (best is None or us < best["us"]): best = dict(cfg=f"{c.tile_m}x{c.tile_n} c{c.cluster_m} dyn{int(c.is_dynamic_persistent)}", err=err, us=us)
        except Exception as e: rec.setdefault("p_fails", []).append(f"{type(e).__name__}: {str(e).strip().splitlines()[-1][-160:]}")
    rec.update(p_best=best, speedup=round(rec["v_us"] / best["us"], 3) if best else None)
    out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps(rec), flush=True)
