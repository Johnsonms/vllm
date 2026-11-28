"""W21 step 3: QuACK MXFP8 kernels vs vLLM's block-128 FP8 path (DeepGEMM + per_token_group_quant + silu_and_mul) on B200.
usage: bench.py MODEL_SNAPSHOT_DIR LAYER OUT.jsonl [Ms]
Real layer weights. vLLM arm = the functions DeepGemmFp8BlockScaledMMKernel calls. QuACK arm = weights RE-QUANTIZED to MXFP8
(1x32 UE8M0) from the dequantized checkpoint. CUDA graph, cold L2, CUPTI, median us. Errors are vs an fp32 matmul on the
dequantized checkpoint weights with the same bf16 activations."""
import glob, json, sys, time, torch, torch.nn.functional as F
from safetensors import safe_open
from flashinfer.testing import bench_gpu_time_with_cupti
import vllm._custom_ops  # noqa
import vllm.envs as envs
from vllm.model_executor.kernels.linear.scaled_mm import deep_gemm as _dgk  # registers fp8_gemm_nt_op  # noqa
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
from quack import gemm_interface as gi
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3

snap, layer, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
Ms = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 8, 32, 128, 512, 2048, 8192]
dev = "cuda"; E8 = is_deep_gemm_e8m0_used()
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{snap}/{idx[name]}", "pt", device=dev) as f: return f.get_tensor(name)
def wpair(names):
    p = f"model.layers.{layer}."
    return torch.cat([load(p + n + ".weight") for n in names]), torch.cat([load(p + n + ".weight_scale_inv") for n in names])
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    quant = QuantFP8(static=False, group_shape=GroupShape(1, 128), use_ue8m0=E8,
                     tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES, column_major_scales=True)
def t(fn):
    ms = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=60)
    return round(1e3 * float(torch.tensor(ms).median()), 2)
def rel(a, ref): return round(((a.float() - ref).norm() / ref.norm()).item(), 5)
out = open(outp, "a")
def emit(**r): out.write(json.dumps(r) + "\n"); out.flush(); print(json.dumps(r), flush=True)
print("E8M0", E8, "tma_aligned", envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES, flush=True)

OPS = {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o": ["self_attn.o_proj"],
       "gate_up": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]}
for op, names in OPS.items():
    wq, ws = wpair(names); ws = ws.float(); N, K = wq.shape
    wdeq = wq.float() * ws.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]   # checkpoint weights, fp32
    W, Ws = deepgemm_post_process_fp8_weight_block(wq.clone(), ws.clone(), (128, 128), E8)       # what vLLM runs
    gated = op == "gate_up"
    wsrc = gi._concat_interleave(wdeq) if gated else wdeq       # QuACK gated wants g0,u0,g1,u1 along N
    B8 = BlockScaledOperand.quantize(wsrc, MXFP8_E4M3).mT
    B8p = BlockScaledOperand.quantize(wdeq, MXFP8_E4M3).mT if gated else B8
    for M in Ms:
        torch.manual_seed(M); x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ wdeq.T
        rec = dict(op=op, N=N, K=K, M=M)
        def step(name, fn, check=None):
            try:
                y = fn(); torch.cuda.synchronize()
                if check is not None: rec[name + "_err"] = check(y)
                rec[name + "_us"] = t(fn)
            except Exception as e:
                rec[name + "_fail"] = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:160]}"
        # ---- vLLM arm
        xq, xs = quant(x)
        def v_gemm():
            o = torch.empty((M, N), dtype=torch.bfloat16, device=dev); torch.ops.vllm.fp8_gemm_nt_op(xq, xs, W, Ws, o, E8); return o
        def v_quant_gemm():
            q, s = quant(x); o = torch.empty((M, N), dtype=torch.bfloat16, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q, s, W, Ws, o, E8); return o
        step("v_quant_in", lambda: quant(x))
        step("v_gemm", v_gemm, lambda y: rel(y, ref))
        step("v_quant_gemm", v_quant_gemm)
        # ---- QuACK arm (MXFP8 re-quantized weights)
        A8 = BlockScaledOperand.quantize(x, MXFP8_E4M3)
        step("q_quant_in", lambda: BlockScaledOperand.quantize(x, MXFP8_E4M3).qdata)
        step("q_gemm", lambda: gi.gemm(A8, B8p), lambda y: rel(y, ref))
        step("q_gemm_qout", lambda: gi.gemm(A8, B8p, out_dtype=MXFP8_E4M3).qdata)
        try: rec["q_gemm_qout_err"] = rel(gi.gemm(A8, B8p, out_dtype=MXFP8_E4M3).dequantize(), ref)
        except Exception as e: pass
        if gated:
            I = N // 2; gref = F.silu(ref[:, :I]) * ref[:, I:]
            def v_act(): 
                o = v_gemm(); return torch.ops._C.silu_and_mul if False else o
            pre = v_gemm()
            def v_silu():
                h = torch.empty((M, I), dtype=torch.bfloat16, device=dev); torch.ops._C.silu_and_mul(h, pre); return h
            def v_mlp_front(with_in_quant, with_out_quant):
                def f():
                    q, s = quant(x) if with_in_quant else (xq, xs)
                    o = torch.empty((M, N), dtype=torch.bfloat16, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q, s, W, Ws, o, E8)
                    h = torch.empty((M, I), dtype=torch.bfloat16, device=dev); torch.ops._C.silu_and_mul(h, o)
                    return quant(h)[0] if with_out_quant else h
                return f
            step("v_silu", v_silu)
            step("v_gemm_silu", v_mlp_front(False, False), lambda y: rel(y, gref))
            step("v_gemm_silu_quant", v_mlp_front(False, True))
            step("v_quant_gemm_silu_quant", v_mlp_front(True, True))
            hq = v_mlp_front(False, False)()
            step("v_quant_h", lambda: quant(hq))
            def q_gated(): return gi.gemm_act(A8, B8, activation="swiglu", store_preact=False)[1]
            step("q_gated", q_gated, lambda y: rel(y, gref))
            step("q_gated_quant", lambda: BlockScaledOperand.quantize(q_gated(), MXFP8_E4M3).qdata)
            step("q_quant_h", lambda: BlockScaledOperand.quantize(hq, MXFP8_E4M3).qdata)
            step("q_gated_vquant", lambda: quant(q_gated())[0])     # hypothetical: QuACK gated + vLLM's 1x128 quant kernel
        emit(**rec)
    del wq, ws, wdeq, W, Ws, B8, B8p; torch.cuda.empty_cache()
