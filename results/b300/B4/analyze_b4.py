"""B4 summary: per cell median speedup over 4 seeds (cuBLAS us / config us), min-max, errors.
usage: python3 analyze_b4.py DIR  -> verify_summary.json + markdown tables on stdout"""
import glob, json, os, statistics as st, sys
d = sys.argv[1]; out = {}; md = []
for s in ("dense_qwen3", "qwen3_moe"):
    cells = {}
    for f in sorted(glob.glob(f"{d}/verify_{s}_s*.jsonl")):
        for l in open(f):
            r = json.loads(l); cells.setdefault((r["n"], r["k"], r["m"]), []).append(r)
    rows = []
    for (n, k, m), rs in sorted(cells.items()):
        sp = [r["cublas_us"] / r["sm100_us"] for r in rs]
        rows.append(dict(n=n, k=k, m=m, cfg=rs[0]["cfg_sm100"], seeds=len(rs),
                         cublas_us=st.median(r["cublas_us"] for r in rs), cfg_us=st.median(r["sm100_us"] for r in rs),
                         speedup=st.median(sp), min=min(sp), max=max(sp),
                         err_cublas=max(r["err_cublas"] for r in rs), err_cfg=max(r["err_sm100"] for r in rs)))
    out[s] = rows
    lo = [r for r in rows if r["speedup"] < 1.10]; reg = [r for r in rows if r["speedup"] < 1.00]
    md.append(f"\n### {s}: {len(rows)} cells, {len({(r['n'],r['k']) for r in rows})} shapes; "
              f"{len(rows)-len(lo)} ≥ 1.10×, {len(lo)} < 1.10×, {len(reg)} < 1.00×; "
              f"speedup range {min(r['speedup'] for r in rows):.2f}–{max(r['speedup'] for r in rows):.2f}×; "
              f"max err cuBLAS {max(r['err_cublas'] for r in rows):.2e}, config {max(r['err_cfg'] for r in rows):.2e}\n")
    md.append("| N,K | M | cuBLAS µs | config µs | speedup (median) | min–max | err cuBLAS | err config | flag |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        flag = "**<1.00**" if r["speedup"] < 1 else ("<1.10" if r["speedup"] < 1.1 else "")
        md.append(f"| {r['n']},{r['k']} | {r['m']} | {r['cublas_us']:.2f} | {r['cfg_us']:.2f} | {r['speedup']:.2f}× | "
                  f"{r['min']:.2f}–{r['max']:.2f} | {r['err_cublas']:.1e} | {r['err_cfg']:.1e} | {flag} |")
json.dump(out, open(f"{d}/verify_summary.json", "w"), indent=1)
print("\n".join(md))
