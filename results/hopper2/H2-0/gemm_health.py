"""30 s BF16 GEMM throughput check on one GPU (run one process per GPU).

usage: CUDA_VISIBLE_DEVICES=<i> python gemm_health.py <label> [seconds]
Prints one JSON line: TFLOPS per 1 s window (min/median/max) and overall.
"""
import json, sys, time
import torch

label = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
N = 8192
torch.manual_seed(0)
a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
c = torch.empty(N, N, device="cuda", dtype=torch.bfloat16)
flop = 2 * N**3
for _ in range(20):  # warmup
    torch.matmul(a, b, out=c)
torch.cuda.synchronize()

ITERS = 50  # one timed chunk ~0.6 s on H100
windows, total_iters = [], 0
t_start = time.perf_counter()
while time.perf_counter() - t_start < secs:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        torch.matmul(a, b, out=c)
    e.record()
    e.synchronize()
    ms = s.elapsed_time(e)
    windows.append(flop * ITERS / (ms * 1e-3) / 1e12)
    total_iters += ITERS
wall = time.perf_counter() - t_start
w = sorted(windows)
print(json.dumps({
    "gpu": label, "shape": f"{N}x{N}x{N} bf16", "seconds": round(wall, 1),
    "tflops_overall": round(flop * total_iters / wall / 1e12, 1),
    "tflops_min": round(w[0], 1), "tflops_median": round(w[len(w) // 2], 1),
    "tflops_max": round(w[-1], 1), "tflops_last5_mean": round(sum(windows[-5:]) / 5, 1),
}))
