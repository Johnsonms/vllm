# Count fused SiLU+group-128 quant producer calls and QuantizedActivation inputs to the
# CUTLASS block-FP8 linear, in-process (V1 multiprocessing off), real weights, eager.
import collections, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import vllm.model_executor.layers.fusion.fused_act_quant as faq
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
from vllm.model_executor.kernels.linear.scaled_mm import cutlass as cm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
cnt = collections.Counter()
key = (SiluAndMul, kFp8Dynamic128Sym)
print("registry has (SiluAndMul, kFp8Dynamic128Sym):", key in faq._FUSED_ACT_QUANT)
if key in faq._FUSED_ACT_QUANT:
    orig = faq._FUSED_ACT_QUANT[key]
    def wrapped(x, linear):
        cnt["producer_calls"] += 1
        cnt["producer_N_K=" + str(tuple(linear.weight.shape))] += 1
        return orig(x, linear)
    faq._FUSED_ACT_QUANT[key] = wrapped
orig_aw = cm.CutlassFp8BlockScaledMMKernel.apply_weights
def aw(self, layer, x, bias=None, **kw):
    cnt["cutlass_block_apply"] += 1
    if isinstance(x, QuantizedActivation):
        cnt["cutlass_block_apply_prequant"] += 1
    return orig_aw(self, layer, x, bias, **kw)
cm.CutlassFp8BlockScaledMMKernel.apply_weights = aw
orig_mfaq = faq.maybe_fused_act_quant
from vllm.model_executor.models import qwen3_moe
def mfaq(act_fn, x, linear):
    cnt["maybe_fused_act_quant key=" + str(get_input_quant_key(linear))] += 1
    return orig_mfaq(act_fn, x, linear)
qwen3_moe.maybe_fused_act_quant = mfaq
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen3.6-35B-A3B-FP8", enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85,
          language_model_only=True)
cnt.clear()
out = llm.generate(["The capital of France is"] * 4, SamplingParams(temperature=0, max_tokens=16))
print("TEXT:", repr(out[0].outputs[0].text))
for k, v in sorted(cnt.items()):
    if v: print(f"COUNT {k}: {v}")
