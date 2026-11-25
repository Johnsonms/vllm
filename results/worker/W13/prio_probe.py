"""Does a stream's priority reach the kernel nodes of a captured torch CUDA graph? usage: prio_probe.py PRIO"""
import ctypes, sys, collections, torch
prio = int(sys.argv[1])
rt = ctypes.CDLL([l.split()[-1] for l in open("/proc/self/maps") if "libcudart" in l][0]) if any("libcudart" in l for l in open("/proc/self/maps")) else None
torch.zeros(1, device="cuda")
if rt is None:
    rt = ctypes.CDLL([l.split()[-1] for l in open("/proc/self/maps") if "libcudart" in l][0])
lo, hi = ctypes.c_int(), ctypes.c_int(); rt.cudaDeviceGetStreamPriorityRange(ctypes.byref(lo), ctypes.byref(hi))
print("device priority range least", lo.value, "greatest", hi.value)
aux = torch.cuda.Stream(priority=prio)
def sp(s):
    p = ctypes.c_int(99); rt.cudaStreamGetPriority(ctypes.c_void_p(s.cuda_stream), ctypes.byref(p)); return p.value
x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16); w = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
def step():
    cur = torch.cuda.current_stream()
    aux.wait_stream(cur)
    with torch.cuda.stream(aux):
        a = torch.nn.functional.silu(x) * 2       # aux-stream kernels
    b = torch.relu(x) + 1                         # main-stream kernels
    cur.wait_stream(aux)
    return a + b
step(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph(keep_graph=True)
cap = torch.cuda.Stream()
with torch.cuda.graph(g, stream=cap):
    step()
print("stream priorities: capture", sp(cap), "aux", sp(aux))
raw = ctypes.c_void_p(int(g.raw_cuda_graph()) if not hasattr(g.raw_cuda_graph(), "value") else g.raw_cuda_graph().value)
n = ctypes.c_size_t(0); rt.cudaGraphGetNodes(raw, None, ctypes.byref(n))
nodes = (ctypes.c_void_p * n.value)(); rt.cudaGraphGetNodes(raw, nodes, ctypes.byref(n))
cnt = collections.Counter()
for nd in nodes:
    t = ctypes.c_int(); rt.cudaGraphNodeGetType(ctypes.c_void_p(nd), ctypes.byref(t))
    if t.value != 0: cnt[("type", t.value)] += 1; continue
    buf = (ctypes.c_int * 16)(); rc = rt.cudaGraphKernelNodeGetAttribute(ctypes.c_void_p(nd), 8, buf)
    cnt[("kernel node priority", buf[0], "rc", rc)] += 1
print(sorted(cnt.items(), key=str))
g.replay(); torch.cuda.synchronize(); print("replay ok")
