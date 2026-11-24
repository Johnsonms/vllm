"""B2 verify on SM103: cuBLAS vs proposed SM103 config vs W11's SM100 config.
Interleaved arms (order rotates per round), 3 rounds, CUPTI + CUDA graph + cold L2.
Per arm the middle of 3 rounds is recorded, as in the manager's verify_skinny_reference.py.
usage: python verify_sm103.py PROPOSED.json SM100_VERIFIED.json OUT.jsonl SEED
  JSON format: {"N,K": {"M": "SkinnyGemmConfig(...)"}}
"""

import json
import sys

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm as sg,
)


def parse(path):
    out = {}
    for key, plans in json.load(open(path)).items():
        n, k = map(int, key.split(","))
        out[(n, k)] = {int(m): eval(c, {"SkinnyGemmConfig": SkinnyGemmConfig})
                       for m, c in plans.items()}
    return out


def t(fn, *args):
    ms = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, cold_l2_cache=True,
                                   repeat_time_ms=100, input_args=args)
    return 1e3 * float(torch.tensor(ms).median())


def main():
    prop, sm100 = parse(sys.argv[1]), parse(sys.argv[2])
    out, seed = sys.argv[3], int(sys.argv[4])
    torch.manual_seed(seed)
    fout = open(out, "a")
    cells = sorted({(n, k, m) for d in (prop, sm100) for (n, k), p in d.items() for m in p})
    for n, k, m in cells:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
        ref = F.linear(x.float(), w.float())
        refmax = ref.abs().max()
        cfgs = {"b300": prop.get((n, k), {}).get(m), "sm100": sm100.get((n, k), {}).get(m)}
        fns = {"cublas": F.linear}
        errs = {"cublas": ((F.linear(x, w).float() - ref).abs().max() / refmax).item()}
        for name, cfg in cfgs.items():
            if cfg is None:
                continue
            fns[name] = (lambda a, b, c=cfg: sg(a, b, c))
            errs[name] = ((sg(x, w, cfg).float() - ref).abs().max() / refmax).item()
        names = list(fns)
        runs = {a: [] for a in names}
        for rnd in range(3):
            r = (rnd + seed) % len(names)
            for a in names[r:] + names[:r]:
                runs[a].append(t(fns[a], x, w))
        rec = dict(n=n, k=k, m=m, seed=seed,
                   cfg_b300=repr(cfgs["b300"]) if cfgs["b300"] else None,
                   cfg_sm100=repr(cfgs["sm100"]) if cfgs["sm100"] else None,
                   **{f"{a}_us": sorted(v)[1] for a, v in runs.items()},
                   **{f"{a}_all": v for a, v in runs.items()},
                   **{f"err_{a}": e for a, e in errs.items()})
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        print({k_: v for k_, v in rec.items() if not k_.endswith("_all")}, flush=True)


if __name__ == "__main__":
    main()
