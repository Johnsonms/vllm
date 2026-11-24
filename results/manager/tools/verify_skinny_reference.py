"""Verify proposed SM100 skinny-GEMM plan entries vs the current dispatch (plan or
cuBLAS). Interleaved arms, 3 rounds, CUPTI + CUDA graph + cold L2.
usage: python verify_skinny.py OUT.jsonl SEED
"""

import json
import sys

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig as C,
    shape_dynamic_skinny_gemm as sg,
)
from vllm.models.qwen4_exp.nvidia.low_latency_gemm import QWEN4_EXP_SM100_GEMM_PLANS

PROPOSED = {
    (512, 2560): {
        2: C(2, 128, 2, vector_width=4, static_k=2560),
        4: C(4, 160, 1, static_k=2560),
        8: C(8, 128, 1, vector_width=4, static_k=2560),
        16: C(16, 64, 1, static_k=2560),
    },
    (62080, 2560): {
        1: C(1, 160, 2, k_unroll=4, vector_width=4),
        2: C(2, 160, 4, k_unroll=2),
        4: C(4, 32, 2, k_unroll=4, vector_width=4),
    },
    (24, 2560): {
        1: C(1, 128, 2, vector_width=4, static_k=2560),
        16: C(16, 160, 1, static_k=2560),
    },
}


def t(fn, *args):
    ms = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True,
                                   repeat_time_ms=100, input_args=args)
    return 1e3 * float(torch.tensor(ms).median())


def main():
    out, seed = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(seed)
    fout = open(out, "a")
    for (n, k), plans in PROPOSED.items():
        for m, new in plans.items():
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
            cur = QWEN4_EXP_SM100_GEMM_PLANS.get((n, k), {}).get(m)
            ref = F.linear(x.float(), w.float())
            y_new = sg(x, w, new).float()
            err_new = ((y_new - ref).abs().max() / ref.abs().max()).item()
            y_lin = F.linear(x, w).float()
            err_lin = ((y_lin - ref).abs().max() / ref.abs().max()).item()

            def cur_fn(a, b):
                return sg(a, b, cur) if cur is not None else F.linear(a, b)

            def new_fn(a, b):
                return sg(a, b, new)

            arms = {"cur": [], "new": []}
            for rnd in range(3):
                order = ("cur", "new") if (rnd + seed) % 2 == 0 else ("new", "cur")
                for a in order:
                    arms[a].append(t(cur_fn if a == "cur" else new_fn, x, w))
            rec = dict(n=n, k=k, m=m, seed=seed, cur="plan" if cur else "cublas",
                       cur_us=sorted(arms["cur"])[1], new_us=sorted(arms["new"])[1],
                       err_new=err_new, err_cublas=err_lin)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(rec, flush=True)


if __name__ == "__main__":
    main()
