import json,sys
for f in sys.argv[1:]:
    for l in open(f):
        r=json.loads(l); b=r["best_us"]; c=r["current_us"]
        print(f"M={r['m']:<3} ({r['n']},{r['k']}) cublas {r['cublas_us']:.2f} | plan {('%.2f'%c) if c else '-'} | best {('%.2f'%b) if b else '-'} x{(r['cublas_us']/b) if b else 0:.2f} {(r['best_cfg'] or '')[16:]}")
