"""Does aux-stream priority survive CUDA-graph capture/replay on this torch/driver?
Main stream: one SM-filling GEMM (8192^3 bf16). Aux stream (waits on an event recorded before the GEMM):
5 small kernels. We measure when the aux kernels start/finish relative to the GEMM, from torch-profiler
kernel timestamps, in 5 modes:
  eager-default, eager-high, graph-default, graph-high (torch instantiate),
  graph-high-nodeprio (same captured graph, instantiated by us with cudaGraphInstantiateFlagUseNodePriority).
Also reads the per-node priority launch attribute from the captured graph."""
import statistics as st, torch
from torch.profiler import profile, ProfilerActivity
from cuda.bindings import runtime as rt

def chk(r):
    err = r[0] if isinstance(r, tuple) else r
    assert int(err) == 0, r
    return r[1:] if isinstance(r, tuple) and len(r) > 1 else None

lo, hi = torch.cuda.Stream.priority_range()
print("torch", torch.__version__, "priority range (least, greatest):", (lo, hi))
A = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); B = torch.randn_like(A); C = torch.empty_like(A)
x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16); ys = [torch.empty_like(x) for _ in range(5)]

def body(aux):
    ev = torch.cuda.Event(); ev.record()
    torch.mm(A, B, out=C)
    with torch.cuda.stream(aux):
        ev.wait(aux)
        for y in ys: torch.mul(x, 1.0001, out=y)
    torch.cuda.current_stream().wait_stream(aux)

def measure(run, n=20):
    for _ in range(3): run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n): run()
        torch.cuda.synchronize()
    ks = sorted([(e.time_range.start, e.time_range.end, e.name) for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA], key=lambda t: t[0])
    gem = [k for k in ks if "mul" not in k[2].lower() or "gemm" in k[2].lower() or "nvjet" in k[2]]
    gem = [k for k in ks if k[1] - k[0] > 100]   # the GEMM is the only kernel > 100 us
    res = []
    for g in gem:
        aux = [k for k in ks if k[1] - k[0] <= 100 and g[0] - 5 <= k[0] <= g[1] + 50][:5]
        if len(aux) == 5: res.append((aux[-1][1] - g[0], g[1] - g[0], aux[0][0] - g[0]))
    return (st.median(r[0] for r in res), st.median(r[1] for r in res), st.median(r[2] for r in res), len(res))

out = {}
for name, prio in (("default", 0), ("p-1", -1), ("high", hi)):
    aux = torch.cuda.Stream(priority=prio)
    out[f"eager-{name}"] = measure(lambda: body(aux))
    g = torch.cuda.CUDAGraph(keep_graph=True)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        body(aux); torch.cuda.synchronize()
        g.capture_begin(); body(aux); g.capture_end()
    g.instantiate()
    out[f"graph-{name}"] = measure(g.replay)
    # node priority attribute of the captured kernel nodes
    graph = g.raw_cuda_graph()
    nodes, n = chk(rt.cudaGraphGetNodes(graph, None))[0:2] if False else (None, None)
    _, _, num = rt.cudaGraphGetNodes(graph, None) if False else (0, 0, 0)
    err, nodes, num = rt.cudaGraphGetNodes(graph, 64)
    prios = []
    for nd in nodes[:num]:
        err, t = rt.cudaGraphNodeGetType(nd)
        if t == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            err, v = rt.cudaGraphKernelNodeGetAttribute(nd, rt.cudaLaunchAttributeID.cudaLaunchAttributePriority)
            prios.append(v.priority if int(err) == 0 else f"err{int(err)}")
    print(f"captured graph ({name} aux): kernel-node priority attributes = {prios}")
    if name == "high":
        err, gexec = rt.cudaGraphInstantiateWithFlags(graph, rt.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagUseNodePriority)
        assert int(err) == 0, err
        def launch(): chk(rt.cudaGraphLaunch(gexec, torch.cuda.current_stream().cuda_stream))
        out["graph-high-nodeprio"] = measure(launch)
print("\n| mode | aux done after GEMM start (us) | GEMM dur (us) | first aux start (us) | samples |\n|---|---|---|---|---|")
for k, (a, gd, f, n) in out.items(): print(f"| {k} | {a:.1f} | {gd:.1f} | {f:.1f} | {n} |")
