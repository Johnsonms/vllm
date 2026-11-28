import json, glob, statistics as st, sys
R="/scratch/kopt-w23"
print("| workload | metric | pass 1 base | pass 1 mx | pass 1 delta | pass 2 base | pass 2 mx | pass 2 delta | mean delta | rounds mx faster (of 12) |\n|---|---|---|---|---|---|---|---|---|---|")
for W in ("d1","d8","d32","d128","p1","p4","p16"):
    key="median_tpot_ms" if W[0]=="d" else "median_ttft_ms"; cells=[]; ds=[]; wins=0; n=0
    for p in ("pass1","pass2"):
        v={a:{} for a in ("base","mx")}
        for a in v:
            for f in glob.glob(f"{R}/v3_{p}/{a}_{W}_k*.json"): v[a][f.split("_k")[-1][:-5]]=json.load(open(f))[key]
        ks=sorted(set(v["base"])&set(v["mx"]))
        if not ks: cells+=["-","-","-"]; continue
        b=st.mean(v["base"][k] for k in ks); m=st.mean(v["mx"][k] for k in ks); d=100*(m/b-1); ds.append(d)
        wins+=sum(v["mx"][k]<v["base"][k] for k in ks); n+=len(ks)
        cells+=[f"{b:.3f}",f"{m:.3f}",f"{d:+.2f} %"]
    name={"d":"decode c","p":"prefill 4096/1 c"}[W[0]]+W[1:]
    print(f"| {name} | {'TPOT' if W[0]=='d' else 'TTFT'} ms | "+" | ".join(cells)+f" | {st.mean(ds):+.2f} % | {wins}/{n} |" if ds else name)
