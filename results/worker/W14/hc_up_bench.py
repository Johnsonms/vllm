"""Flash-Next hyper-connection up projection + gate mix at small M: unfused (cuBLAS F.linear + hc_gate_mix) vs the existing
fused op that main uses only from M = 48. CUDA graph, cold L2, CUPTI median us. Also checks the outputs agree."""
import json, sys, torch, torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti
import vllm._custom_ops  # noqa
from vllm.models.qwen4_exp.nvidia.ops import hc
def t(fn, *args):
    ms = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=50, input_args=args)
    return 1e3 * float(torch.tensor(ms).median())
torch.manual_seed(0); w = torch.randn(10240, 320, device="cuda", dtype=torch.bfloat16) / 320**0.5
print("fused op present:", hasattr(torch.ops._C, "qwen4_exp_hc_up_gate_mix"), "| MIN_FUSED_UP_M =", hc.MIN_FUSED_UP_M)
for m in (1, 2, 4, 8, 16, 32, 48):
    low = torch.randn(m, 320, device="cuda", dtype=torch.bfloat16); xn = torch.randn(m, 10240, device="cuda", dtype=torch.bfloat16)
    unf = lambda a, b, c: hc.hc_gate_mix(b, F.linear(a, c), 4)
    fus = lambda a, b, c: hc.hc_up_gate_mix(a, b, c)
    y0, y1 = unf(low, xn, w), fus(low, xn, w); torch.cuda.synchronize()
    err = (y0.float() - y1.float()).abs().max().item(); same = bool((y0 == y1).all())
    rec = dict(m=m, unfused_us=round(t(unf, low, xn, w), 2), gemm_only_us=round(t(F.linear, low, w), 2), fused_us=round(t(fus, low, xn, w), 2), max_abs_diff=err, bit_identical=same)
    print(json.dumps(rec), flush=True)
