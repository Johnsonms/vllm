"""Which all-reduce does vLLM's unfused path dispatch at TP=N, per tensor size, and how long does it take?
Compares, in a CUDA graph per size (hidden H, bf16):
  std        : tensor_model_parallel_all_reduce (whatever the device communicator picks) + vLLM RMSNorm(residual)
  fused_mnnvl: flashinfer allreduce_fusion (mnnvl, AR+residual+RMSNorm) via vLLM's fused-op helper if present
usage: torchrun --nproc-per-node=8 ar_probe.py H tokens...
"""
import os, sys, json
import torch, torch.distributed as dist
from vllm.config.vllm import VllmConfig, CompilationConfig, ParallelConfig, set_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (graph_capture, init_distributed_environment,
                                             initialize_model_parallel, get_tp_group)

H = int(sys.argv[1]); toks = [int(t) for t in sys.argv[2:]]
rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
torch.accelerator.set_device_index(dev); torch.set_default_device(dev)
cfg = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=ws))
with set_current_vllm_config(cfg):
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=ws)
    dc = get_tp_group().device_communicator
    if rank == 0:
        print("comms:", {k: (getattr(dc, k) is not None and not getattr(getattr(dc, k), 'disabled', False))
                         for k in ("ca_comm", "fi_ar_comm", "symm_mem_comm", "pynccl_comm", "qr_comm")}, flush=True)
    from vllm.model_executor.layers.layernorm import RMSNorm
    norm = RMSNorm(H).to(dev, torch.bfloat16)

    def which(x):
        if dc.fi_ar_comm is not None and not dc.fi_ar_comm.disabled and dc.fi_ar_comm.should_use_fi_ar(x):
            return "flashinfer"
        if dc.ca_comm is not None and not dc.ca_comm.disabled and dc.ca_comm.should_custom_ar(x):
            return "custom_ar"
        if dc.symm_mem_comm is not None and dc.symm_mem_comm.should_use_symm_mem(x):
            return "symm_mem"
        return "pynccl"

    res = []
    for n in toks:
        x = torch.randn(n, H, dtype=torch.bfloat16); r = torch.randn(n, H, dtype=torch.bfloat16)
        w = which(x)
        def step():
            y = tensor_model_parallel_all_reduce(x)
            return norm(y, r)
        for _ in range(3): step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with graph_capture(device=dev) as gc, torch.cuda.graph(g, stream=gc.stream):
            for _ in range(10): step()
        for _ in range(5): g.replay()
        torch.cuda.synchronize(); dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(20):
            s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / 10)
        ts.sort(); med = ts[len(ts) // 2]
        if rank == 0:
            rec = dict(H=H, tokens=n, MB=round(x.nbytes / 2**20, 2), backend=w, std_ms=round(med, 4))
            print(json.dumps(rec), flush=True); res.append(rec)
dist.barrier()
