"""W19 knob script (torch only): which difference from repro.py makes cuMulticastBindMem block when one GPU has no active NVLink?
usage: CUDA_VISIBLE_DEVICES=a,b,c,d python repro2.py NPROC
knobs (environment):
  SIZES_MB   comma list, one rendezvous per entry, in order, buffers kept alive   (default "1"; vLLM TP4 does "32,96")
  GROUP      world | sub     sub = dist.new_group(all ranks), like vLLM's 'tp:0'  (default world)
  WARM_NCCL  1 = one NCCL all_reduce on the group before the first rendezvous      (default 0)
  STAGGER    seconds; rank r sleeps r*STAGGER before each rendezvous               (default 0)
  DEADLINE   seconds for the whole run; the parent then reports and kills          (default 90)
  plus torch's own: TORCH_SYMM_MEM_DISABLE_MULTICAST=1, TORCH_SYMMMEM_IMPLICIT_POOL=0, TORCH_CPP_LOG_LEVEL=INFO
Each rank prints a line before and after every step, so the last line per rank shows where it stopped."""
import os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

def say(rank, msg):
    print(f"[rank {rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)

def worker(rank, n, port):
    import torch.distributed._symmetric_memory as sm
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=n, device_id=dev)
    group = dist.group.WORLD
    if os.environ.get("GROUP", "world") == "sub":
        group = dist.new_group(list(range(n)), backend="nccl")
    say(rank, f"group {group.group_name}")
    if os.environ.get("WARM_NCCL", "0") == "1":
        x = torch.ones(8, device=dev); dist.all_reduce(x, group=group); say(rank, f"nccl all_reduce ok ({int(x[0])})")
    keep = []
    for i, mb in enumerate(float(s) for s in os.environ.get("SIZES_MB", "1").split(",")):
        time.sleep(rank * float(os.environ.get("STAGGER", 0)))
        nbytes = int(mb * (1 << 20))
        say(rank, f"rendezvous {i} ({mb:g} MiB): empty")
        buf = sm.empty(nbytes, dtype=torch.uint8, device=dev)
        say(rank, f"rendezvous {i} ({mb:g} MiB): enter")
        t = time.time()
        try:
            h = sm.rendezvous(buf, group.group_name)
            say(rank, f"rendezvous {i} ({mb:g} MiB): ok in {time.time()-t:.2f}s multicast_ptr={'nonzero' if h.multicast_ptr else 0}")
            keep.append((buf, h))
        except RuntimeError as e:
            say(rank, f"rendezvous {i} ({mb:g} MiB): RuntimeError after {time.time()-t:.2f}s: {str(e).splitlines()[0][:160]}")
    dist.barrier(group=group)
    say(rank, "done")

if __name__ == "__main__":
    n = int(sys.argv[1]); port = int(os.environ.get("PORT", 29500 + os.getpid() % 500))
    print("knobs:", {k: os.environ.get(k) for k in ("SIZES_MB", "GROUP", "WARM_NCCL", "STAGGER", "TORCH_SYMM_MEM_DISABLE_MULTICAST", "TORCH_SYMMMEM_IMPLICIT_POOL")}, flush=True)
    ps = [mp.get_context("spawn").Process(target=worker, args=(r, n, port)) for r in range(n)]
    [p.start() for p in ps]
    t0 = time.time(); deadline = float(os.environ.get("DEADLINE", 90))
    while any(p.is_alive() for p in ps) and time.time() - t0 < deadline: time.sleep(1)
    alive = [i for i, p in enumerate(ps) if p.is_alive()]
    if alive:
        print(f"RESULT: HUNG ranks {alive} after {deadline:.0f}s (last line of each rank above shows the step)", flush=True)
        [ps[i].kill() for i in alive]; [p.join() for p in ps]
    else:
        print(f"RESULT: {'OK' if all(p.exitcode == 0 for p in ps) else 'FAILED'} exit codes {[p.exitcode for p in ps]} in {time.time()-t0:.0f}s", flush=True)
    tmp = os.environ.get("TMPDIR", "/tmp")
    for p in ps:  # remove only the socket files of this run's own (now dead) processes
        try: os.unlink(os.path.join(tmp, f"symm_mem-{p.pid}"))
        except OSError: pass
