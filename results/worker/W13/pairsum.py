import json,statistics as st,sys
roots=sys.argv[1:]; allm={}
for root in roots:
    for C in (1,2,4,8,16,32,64):
        d=[];A=[];B=[]
        for K in range(1,7):
            try: a=json.load(open(f"{root}/a_c{C}_k{K}.json"))["median_tpot_ms"]; b=json.load(open(f"{root}/b_c{C}_k{K}.json"))["median_tpot_ms"]
            except Exception: continue
            d.append((b/a-1)*100);A.append(a);B.append(b)
        if not d: continue
        allm.setdefault(C,[]).append(st.mean(d))
        print(root,f"c{C}: a {st.median(A):.3f} b {st.median(B):.3f} | diff mean {st.mean(d):+.2f} min {min(d):+.2f} max {max(d):+.2f} neg {sum(x<0 for x in d)}/{len(d)}")
print("mean of passes:",{C:round(st.mean(v),2) for C,v in allm.items()})
