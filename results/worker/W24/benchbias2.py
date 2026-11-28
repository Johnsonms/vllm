"""Does my bench-style DeepGEMM call launch extra kernels that serving does not? List kernels per call."""
import json, sys, torch, collections
from safetensors import safe_open
import vllm._custom_ops  # noqa
import vllm.envs as envs
from vllm.model_executor.kernels.linear.scaled_mm import deep_gemm as _dgk  # noqa
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import deepgemm_post_process_fp8_weight_block
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
from vllm.config import VllmConfig, set_current_vllm_config
snap = sys.argv[1]; dev = "cuda"; E8 = is_deep_gemm_e8m0_used()
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{snap}/{idx[name]}", "pt", device=dev) as f: return f.get_tensor(name)
p = "model.layers.10.mlp.down_proj."
wq = load(p + "weight"); ws = load(p + "weight_scale_inv").float()
W, Ws = deepgemm_post_process_fp8_weight_block(wq, ws, (128, 128), E8)
print("weight scale after post-process:", Ws.dtype, tuple(Ws.shape), flush=True)
with set_current_vllm_config(VllmConfig()):
    quant = QuantFP8(static=False, group_shape=GroupShape(1, 128), use_ue8m0=E8, tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES, column_major_scales=True)
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8_packed_for_deepgemm as PQ
M = 32; x = torch.randn(M, wq.shape[1], device=dev, dtype=torch.bfloat16)
def call():
    q, s = PQ(x, 128, use_ue8m0=True); o = torch.empty((M, wq.shape[0]), dtype=torch.bfloat16, device=dev); torch.ops.vllm.fp8_gemm_nt_op(q, s, W, Ws, o, E8); return o
for _ in range(3): call()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(20): call()
    torch.cuda.synchronize()
c = collections.defaultdict(lambda: [0, 0])
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA and "Memcpy" not in e.name and "Memset" not in e.name: c[e.name[:80]][0] += 1; c[e.name[:80]][1] += e.device_time
for k, (n, us) in sorted(c.items(), key=lambda x: -x[1][1]): print(f"{n/20:5.1f}/call {us/n:7.1f} us  {k}")
