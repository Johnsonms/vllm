# Count input-norm fused quant calls by key, and pre-quantized inputs to the CUTLASS
# block-FP8 linear, in-process (V1 multiprocessing off), real weights, eager.
import collections, os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.kernels.linear.scaled_mm import cutlass as cm
import vllm.models.qwen3_5.nvidia.model as qm
cnt = collections.Counter()
orig = qm.gemma_add_rms_norm_quant
def wrapped(*a, **kw):
    key = kw.get("key", a[5] if len(a) > 5 else None)
    cnt[f"norm_quant key={key}"] += 1
    return orig(*a, **kw)
qm.gemma_add_rms_norm_quant = wrapped
orig_aw = cm.CutlassFp8BlockScaledMMKernel.apply_weights
def aw(self, layer, x, bias=None, **kw):
    cnt["cutlass_block_apply"] += 1
    if isinstance(x, QuantizedActivation):
        cnt["cutlass_block_apply_prequant N_K=" + str(tuple(layer.weight.shape))] += 1
    return orig_aw(self, layer, x, bias, **kw)
cm.CutlassFp8BlockScaledMMKernel.apply_weights = aw
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen3.6-35B-A3B-FP8", enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85,
          language_model_only=True)
cnt.clear()
out = llm.generate(["The capital of France is"] * 4, SamplingParams(temperature=0, max_tokens=16))
print("TEXT:", repr(out[0].outputs[0].text))
for k, v in sorted(cnt.items()):
    if v: print(f"COUNT {k}: {v}")
