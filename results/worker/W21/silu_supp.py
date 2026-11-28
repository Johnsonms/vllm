"""Supplement: serving runs silu*mul as an Inductor (Triton) kernel, not the _C.silu_and_mul custom op. Time both, same harness."""
import json, torch, torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti
import vllm._custom_ops  # noqa
def t(fn):
    return round(1e3 * float(torch.tensor(bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=60)).median()), 2)
for I in (25600,):
    comp = torch.compile(lambda o: F.silu(o[:, :I]) * o[:, I:], dynamic=False)
    for M in (1, 8, 32, 128, 512, 2048, 8192):
        o = torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16)
        def custom():
            h = torch.empty((M, I), dtype=torch.bfloat16, device="cuda"); torch.ops._C.silu_and_mul(h, o); return h
        torch._dynamo.reset(); comp = torch.compile(lambda o: F.silu(o[:, :I]) * o[:, I:], dynamic=False); comp(o); comp(o)
        print(json.dumps(dict(I=I, M=M, custom_op_us=t(custom), inductor_us=t(lambda: comp(o)))), flush=True)
