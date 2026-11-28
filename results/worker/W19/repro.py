"""Standalone symm-mem rendezvous repro (torch only). usage: repro.py NPROC [stale_rank]
With stale_rank set, that rank pre-creates $TMPDIR/symm_mem-<its pid> (what a SIGKILLed earlier
process with the same pid leaves behind) and swallows RuntimeError, like vLLM's _init_mnnvl_buffer."""
import faulthandler, os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, n, port, stale):
    faulthandler.dump_traceback_later(40, exit=False)
    import torch.distributed._symmetric_memory as sm
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=n)
    if stale == rank:
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"symm_mem-{os.getpid()}")
        open(path, "w").close()
        print(f"[rank {rank}] created stale {path}", flush=True)
    t = time.time()
    try:
        buf = sm.empty(1 << 20, dtype=torch.uint8, device=dev)
        h = sm.rendezvous(buf, dist.group.WORLD.group_name)
        print(f"[rank {rank}] rendezvous ok in {time.time()-t:.2f}s multicast_ptr={'nonzero' if h.multicast_ptr else 0}", flush=True)
    except RuntimeError as e:
        print(f"[rank {rank}] RuntimeError after {time.time()-t:.2f}s: {str(e).splitlines()[0][:200]}", flush=True)
    dist.barrier()
    print(f"[rank {rank}] barrier done", flush=True)
    time.sleep(float(os.environ.get("HOLD", 0)))

if __name__ == "__main__":
    n = int(sys.argv[1]); stale = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    port = int(os.environ.get("PORT", 29500 + os.getpid() % 500))
    ps = [mp.get_context("spawn").Process(target=worker, args=(r, n, port, stale)) for r in range(n)]
    [p.start() for p in ps]
    t0 = time.time(); deadline = float(os.environ.get("DEADLINE", 60))
    while any(p.is_alive() for p in ps) and time.time() - t0 < deadline: time.sleep(1)
    alive = [p for p in ps if p.is_alive()]
    if alive:
        for p in alive:
            st = open(f"/proc/{p.pid}/stat").read().split(); c0 = int(st[13]) + int(st[14])
            time.sleep(2 / len(alive))
            st = open(f"/proc/{p.pid}/stat").read().split(); c1 = int(st[13]) + int(st[14])
            print(f"HUNG pid-rank {ps.index(p)} cpu_ticks_delta={c1-c0}", flush=True)
        [p.kill() for p in alive]
    print("exit", [p.exitcode for p in ps], f"{time.time()-t0:.0f}s")
