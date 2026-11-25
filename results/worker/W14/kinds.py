"""Launches and time per step by kind. usage: kinds.py kernels.json..."""
import json, re, sys, collections
KINDS = [("cuBLAS BF16 GEMM/GEMV", r"nvjet_|splitKreduce|dot_kernel|reduce_1Block|cutlass_80_|gemv"),
 ("skinny GEMM (CuTe DSL)", r"CuteSkinnyGemm"), ("routed MoE GEMM (bmm)", r"^bmm_"),
 ("MoE routing/activation/finalize", r"moe::dev"), ("block FP8 linear GEMM (cutlass)", r"cutlass_3x_gemm_fp8|scaled_mm|cutlass::device_kernel"),
 ("FP8 group quant", r"per_token_group_quant"), ("all-reduce", r"llreduce|ncclDevKernel"),
 ("hyper-connection (hc_*)", r"_hc_|HcDown|hc_down"), ("GDN (conv, recurrent, norm, gates)", r"gdn|causal_conv1d|gated_delta|_fused_sigmoid_gate"),
 ("attention (fmha, QSA, rope, cache)", r"fmha|_qsa_|rope|reshape_and_cache|indexer|topk|TopK"), ("norm (rms/layer)", r"norm|Norm"),
 ("act_and_mul", r"act_and_mul"), ("torch elementwise/index/copy", r"at::native|elementwise|index|copy|Fill"), ("sampling", r"sampling|SoftMax")]
for f in sys.argv[1:]:
    d = json.load(open(f)); agg = collections.defaultdict(lambda: [0, 0]); other = []
    for k in d["kernels"]:
        for name, rx in KINDS:
            if re.search(rx, k["name"]): agg[name][0] += k["us"]; agg[name][1] += k["calls"]; break
        else: agg["other"][0] += k["us"]; agg["other"][1] += k["calls"]; other.append(k["name"][:60])
    tot = d["summed_kernel_us"]
    print(f"\n{f}: {d['kernel_launches']/32:.0f} launches/step, {tot/32:.0f} us summed/step\n| kind | launches/step | us/step | % of kernel time |\n|---|---|---|---|")
    for name, (us, n) in sorted(agg.items(), key=lambda x: -x[1][0]): print(f"| {name} | {n/32:.0f} | {us/32:.0f} | {100*us/tot:.1f} |")
    if other: print("other:", sorted(set(other))[:12])
