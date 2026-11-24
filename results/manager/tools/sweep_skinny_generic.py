"""Sweep cuteDSL skinny-GEMM configs vs cuBLAS on this GPU for arbitrary (N, K) shapes.

usage: python sweep_skinny_generic.py SHAPES.json OUT.jsonl M [M ...]
  SHAPES.json: [[N, K, "label"], ...]
Current-plan column uses the generic table (vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py).
"""

import itertools
import json
import sys

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    row_stride_ok,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans import get_skinny_gemm_plan



def t(fn, *args):
    """Median CUPTI time in us, CUDA graph, cold L2 (tensors passed as args)."""
    ms = bench_gpu_time_with_cupti(
        fn, use_cuda_graph=True, cold_l2_cache=True, repeat_time_ms=50,
        input_args=args,
    )
    return 1e3 * float(torch.tensor(ms).median())


def main():
    shapes = [tuple(s[:2]) for s in json.load(open(sys.argv[1]))]
    out_path, ms = sys.argv[2], [int(x) for x in sys.argv[3:]]
    torch.manual_seed(0)
    dev = "cuda"
    fout = open(out_path, "a")
    for m in ms:
        for n, k in shapes:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) / k**0.5
            ref = F.linear(x, w)
            base_us = t(F.linear, x, w)
            cur = (get_skinny_gemm_plan(n, k) or {}).get(m)
            rows = []
            grid = itertools.product(
                (32, 64, 128, 160, 256), (1, 2, 4), (1, 2, 4), (4, 8), (None, k)
            )
            for bs, opb, ku, vw, sk in grid:
                if n % opb or k % (bs * vw) and sk is None:
                    continue
                cfg = SkinnyGemmConfig(m, bs, opb, k_unroll=ku, vector_width=vw,
                                       static_k=sk)
                if not row_stride_ok(x, cfg):
                    continue
                try:
                    y = shape_dynamic_skinny_gemm(x, w, cfg)
                    torch.cuda.synchronize()
                except Exception:
                    continue
                err = (y.float() - ref.float()).abs().max().item()
                if err > 0.05:
                    continue
                us = t(lambda a, b, c=cfg: shape_dynamic_skinny_gemm(a, b, c), x, w)
                rows.append((us, cfg, err))
            rows.sort(key=lambda r: r[0])
            cur_us = None
            if cur is not None:
                cur_us = t(lambda a, b: shape_dynamic_skinny_gemm(a, b, cur), x, w)
            best = rows[0] if rows else None
            rec = dict(
                m=m, n=n, k=k, cublas_us=base_us, current_plan=repr(cur),
                current_us=cur_us,
                best_us=best[0] if best else None,
                best_cfg=repr(best[1]) if best else None,
                best_err=best[2] if best else None,
                top3=[(r[0], repr(r[1])) for r in rows[:3]],
            )
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"M={m} ({n},{k}) cublas {base_us:.2f}us cur {cur_us} "
                  f"best {rec['best_us']}us {rec['best_cfg']}", flush=True)


if __name__ == "__main__":
    main()
