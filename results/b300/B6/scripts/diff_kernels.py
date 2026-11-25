"""Per-step kernel time per name across concurrencies: usage diff_kernels.py ARM [N]"""
import json, sys
arm = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 25; STEPS = 32
d = {c: {k["name"]: k for k in json.load(open(f"{arm}/k_c{c}.json"))["kernels"]} for c in (2, 4, 8)}
names = set().union(*d.values())
def us(c, n): return d[c][n]["us"] / STEPS if n in d[c] else 0.0
def calls(c, n): return d[c][n]["calls"] / STEPS if n in d[c] else 0
rows = sorted(names, key=lambda n: -abs(us(4, n) - us(2, n)))
print("| kernel | µs/step c2 | c4 | c8 | Δ c2→c4 | Δ c4→c8 | calls/step c2/c4/c8 |\n|---|---|---|---|---|---|---|")
for n in rows[:N]:
    print(f"| `{n[:95]}` | {us(2,n):.1f} | {us(4,n):.1f} | {us(8,n):.1f} | {us(4,n)-us(2,n):+.1f} | {us(8,n)-us(4,n):+.1f} | {calls(2,n):g}/{calls(4,n):g}/{calls(8,n):g} |")
tot = {c: sum(k["us"] for k in d[c].values()) / STEPS for c in d}
print(f"\nsummed kernel µs/step: c2 {tot[2]:.0f}, c4 {tot[4]:.0f}, c8 {tot[8]:.0f}")
