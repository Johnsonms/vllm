"""B8 step 1: does the SiLU-and-mul + block-FP8 quant fusion fire on SM103?
Loads Qwen3.6-35B-A3B-FP8 in-process, logs the shared-expert down_proj kernel class and
_input_quant_key, then profiles a short greedy decode (4 prompts x 32 tokens) and lists the
act/quant kernels per decode step. usage: PYTHONPATH=<tree> python b8_check.py OUT.json"""
import json, os, sys, collections
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
import torch, vllm
from vllm import LLM, SamplingParams

def inspect_model(model):
    out = {}
    for name, mod in model.named_modules():
        if name.endswith("shared_expert.down_proj"):
            qm = getattr(mod, "quant_method", None)
            lin = getattr(qm, "fp8_linear", None)
            out = dict(module=name, quant_method=type(qm).__name__, fp8_linear=type(lin).__name__,
                       input_quant_key=repr(getattr(mod, "_input_quant_key", None)),
                       kernel_input_quant_key=repr(lin.input_quant_key()) if lin is not None and hasattr(lin, "input_quant_key") else None)
            break
    act = {}
    for name, mod in model.named_modules():
        if name.endswith("shared_expert.act_fn") or name.endswith("shared_expert") :
            act[name.split(".")[-1]] = type(mod).__name__
        if len(act) >= 2: break
    out["shared_expert_types"] = act
    return out

llm = LLM("Qwen/Qwen3.6-35B-A3B-FP8", max_model_len=4096, gpu_memory_utilization=0.85,
          enable_prefix_caching=False, language_model_only=True, max_num_seqs=64)
info = llm.apply_model(inspect_model)[0]
info["vllm_file"] = vllm.__file__
sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
prompts = ["The capital of France is", "def fibonacci(n):", "Explain why the sky is blue.", "1, 1, 2, 3, 5,"]
llm.generate(prompts, sp)  # warm
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    llm.generate(prompts, sp)
    torch.cuda.synchronize()
cnt = collections.Counter(e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
keys = [k for k in cnt if any(s in k.lower() for s in ("quant", "silu", "act_and_mul"))]
info["act_quant_kernels_total_calls"] = {k[:140]: cnt[k] for k in sorted(keys, key=lambda k: -cnt[k])}
json.dump(info, open(sys.argv[1], "w"), indent=1)
print(json.dumps(info, indent=1))
