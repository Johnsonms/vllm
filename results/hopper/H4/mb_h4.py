"""H4: QuACK gemm_w4a16 (int4awq) vs vLLM Machete (uint4 + zero points, group 128) on Qwen3-8B/32B layer shapes.
CUPTI, cold L2, CUDA graph, median of 5. Plus: real-checkpoint AWQ -> QuACK repack (time + correctness)."""
import glob, json, statistics as st, sys, time
import torch, torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti
from safetensors import safe_open
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
from quack.gemm_w4 import gemm_w4a16, prepare_w4_weight

torch.manual_seed(0); dev = "cuda"; G = 128
MS = [1, 8, 16, 32, 64, 128, 512, 2048]
SHAPES = {  # name: (K, N)
    "8b.qkv": (4096, 6144), "8b.o": (4096, 4096), "8b.gate_up": (4096, 24576), "8b.down": (12288, 4096),
    "32b.qkv": (5120, 10240), "32b.o": (8192, 5120), "32b.gate_up": (5120, 51200), "32b.down": (25600, 5120),
}
R = 5


def bench(fn, args):
    return st.median(st.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True,
                                                         input_args=args, repeat_time_ms=100)) for _ in range(R)) * 1e3


def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()


def make_machete(q_nk, s_nk, z_nk, dtype):
    """q/s/z given (N, K)/(N, K/G); Machete wants (K, N) layouts, packed along K, zeros as -s*z."""
    wq = pack_quantized_values_into_int32(q_nk.t().contiguous().int(), scalar_types.uint4, packed_dim=0)  # (K/8, N)
    bq = ops.machete_prepack_B(wq.t().contiguous().t(), a_type=dtype, b_type=scalar_types.uint4, group_scales_type=dtype)
    s = s_nk.t().contiguous().to(dtype)
    zs = (-1.0 * s * z_nk.t().contiguous().to(dtype)).contiguous()
    return bq, s, zs


res = {"shapes": {}, "repack": {}}
for name, (K, N) in SHAPES.items():
    q = torch.randint(0, 16, (N, K), device=dev, dtype=torch.uint8)
    s = (torch.rand(N, K // G, device=dev) * 0.01 + 0.002).to(torch.bfloat16)
    z = torch.randint(0, 16, (N, K // G), device=dev, dtype=torch.uint8)
    W = (q.float() - z.float().repeat_interleave(G, 1)) * s.float().repeat_interleave(G, 1)  # (N, K) fp32 reference
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    t0 = time.time(); blob, sf = prepare_w4_weight(packed, (s, z), wformat="int4awq"); torch.cuda.synchronize(); t_prep = time.time() - t0
    mb, ms_, mz = make_machete(q, s, z, torch.bfloat16)
    mb16, ms16, mz16 = make_machete(q, s, z, torch.float16)
    res["shapes"][name] = {"K": K, "N": N, "quack_prepare_s": t_prep, "M": {}}
    for M in MS:
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        x16 = x.to(torch.float16)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ W.t()
        yq = gemm_w4a16(x, blob, sf, wformat="int4awq")
        ym = ops.machete_mm(a=x, b_q=mb, b_type=scalar_types.uint4, b_group_zeros=mz, b_group_scales=ms_, b_group_size=G)
        ym16 = ops.machete_mm(a=x16, b_q=mb16, b_type=scalar_types.uint4, b_group_zeros=mz16, b_group_scales=ms16, b_group_size=G)
        tq = bench(lambda x: gemm_w4a16(x, blob, sf, wformat="int4awq"), (x,))
        tm = bench(lambda x: ops.machete_mm(a=x, b_q=mb, b_type=scalar_types.uint4, b_group_zeros=mz, b_group_scales=ms_, b_group_size=G), (x,))
        tm16 = bench(lambda x: ops.machete_mm(a=x, b_q=mb16, b_type=scalar_types.uint4, b_group_zeros=mz16, b_group_scales=ms16, b_group_size=G), (x16,))
        r = dict(quack_us=tq, machete_bf16_us=tm, machete_fp16_us=tm16, speedup_vs_bf16=tm / tq, speedup_vs_fp16=tm16 / tq,
                 relerr_quack=rel(yq, ref), relerr_machete_bf16=rel(ym, ref), relerr_machete_fp16=rel(ym16, ref))
        res["shapes"][name]["M"][M] = r
        print(f"{name:12s} M={M:5d}: quack {tq:8.1f}us  machete bf16 {tm:8.1f}us ({tm/tq:.2f}x)  fp16 {tm16:8.1f}us ({tm16/tq:.2f}x)"
              f"  relerr q {r['relerr_quack']:.4f} m {r['relerr_machete_bf16']:.4f}/{r['relerr_machete_fp16']:.4f}", flush=True)
    del W
    torch.cuda.empty_cache()

# ---- real checkpoint: AWQ (K, N/8 int32, order [0,2,4,6,1,3,5,7]) -> QuACK int4awq
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def unpack_awq(t):  # (R, C/8) int32 -> (R, C) int in logical order
    shifts = torch.tensor([4 * AWQ_ORDER.index(i) for i in range(8)], device=t.device, dtype=torch.int32)
    return ((t[..., None] >> shifts) & 0xF).reshape(t.shape[0], -1)


for model in ["Qwen3-8B-AWQ", "Qwen3-32B-AWQ"]:
    files = sorted(glob.glob(f"{__import__('os').environ['HF_HOME']}/hub/models--Qwen--{model}/snapshots/*/*.safetensors"))
    pref = "model.layers.0.mlp.gate_proj"
    for f in files:
        with safe_open(f, "pt", device="cuda") as fh:
            if f"{pref}.qweight" in fh.keys():
                qw, qz, sc = (fh.get_tensor(f"{pref}.{k}") for k in ("qweight", "qzeros", "scales"))
                break
    torch.cuda.synchronize(); t0 = time.time()
    q = unpack_awq(qw).t().contiguous().to(torch.uint8)             # (N, K)
    z = unpack_awq(qz).t().contiguous().to(torch.uint8)             # (N, K/G)
    s = sc.t().contiguous().to(torch.bfloat16)                      # fp16 -> bf16 (precision note)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    blob, sf = prepare_w4_weight(packed, (s, z), wformat="int4awq")
    torch.cuda.synchronize(); t_rep = time.time() - t0
    Wref = awq_dequantize_triton(qw, sc, qz).float()                # (K, N) vLLM's own AWQ dequant
    K, N = Wref.shape
    x = (torch.randn(64, K, device=dev) * 0.5).to(torch.bfloat16)
    ref = x.float() @ Wref
    yq = gemm_w4a16(x, blob, sf, wformat="int4awq")[:, :N]
    res["repack"][model] = dict(layer=pref, K=K, N=N, repack_s=t_rep, relerr_vs_vllm_awq_dequant=rel(yq, ref))
    print(f"{model} {pref} (K={K}, N={N}): AWQ->QuACK repack {t_rep*1e3:.1f} ms (on GPU), relerr vs vLLM awq_dequantize {rel(yq, ref):.4f}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
