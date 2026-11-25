"""Module walk: log every unquantized linear / LM-head layer with its local (N, K), and the
distinct (weight shape, x shape) pairs its apply() sees. usage: walk.py MODEL TP OUTDIR [ep]
The patches sit at module top level so that spawned TP workers (which re-import this file) get them too."""
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
OUTDIR = os.environ.get("WALK_OUTDIR") or sys.argv[3]
os.environ["WALK_OUTDIR"] = OUTDIR

from vllm.model_executor.layers import linear as L, vocab_parallel_embedding as V

seen = set()


def log(rec):
    with open(f"{OUTDIR}/pid{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def patch(cls):
    op, oa = cls.process_weights_after_loading, cls.apply

    def p(self, layer):
        op(self, layer)
        w = layer.weight
        log(dict(ev="load", method=cls.__name__, cls=type(layer).__name__,
                 prefix=getattr(layer, "prefix", None), shape=list(w.shape), dtype=str(w.dtype),
                 bias=getattr(layer, "bias", None) is not None,
                 use_skinny=bool(getattr(layer, "use_skinny_gemm", False))))

    def a(self, layer, x, bias=None):
        key = (cls.__name__, type(layer).__name__, tuple(layer.weight.shape), tuple(x.shape),
               str(x.dtype), bias is not None)
        if key not in seen:
            seen.add(key)
            log(dict(ev="apply", key=key))
        return oa(self, layer, x, bias)

    cls.process_weights_after_loading, cls.apply = p, a


from vllm.model_executor.layers.quantization import fp8 as F8
patch(F8.Fp8LinearMethod)

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    model, tp = sys.argv[1], int(sys.argv[2])
    kw = dict(enable_expert_parallel=True) if len(sys.argv) > 4 else {}
    llm = LLM(model, tensor_parallel_size=tp, load_format="dummy", max_model_len=2048,
              gpu_memory_utilization=0.9, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    llm.generate([{"prompt_token_ids": list(range(1000, 1016))}],
                 SamplingParams(max_tokens=4, temperature=0))
    print("WALK_DONE", flush=True)
    os._exit(0)
