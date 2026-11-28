# Investigate: vLLM TP>1 startup hangs in torch symmetric-memory rendezvous when FABRIC handles are not permitted

## Goal
Find out why vLLM workers spin forever at 100% CPU, with the GPU idle, inside
`torch.distributed._symmetric_memory.rendezvous` during startup on a single-node
8x B200 host where CUDA FABRIC memory handles are not permitted (no IMEX channel).
Then propose the smallest correct fix so vLLM falls back instead of hanging.
Investigation and a proposed patch only: do not push or open anything.

## Environment
- 8x B200 (SM100), single node, all GPUs on one NVLink domain, inside a container.
- No IMEX channel: `/dev/nvidia-caps-imex-channels/channel0` is not available to the container.
- vLLM main at 14e9c30d86 (includes #59208), torch 2.14 (cu13.2), FlashInfer from that venv.
- Model: Qwen/Qwen3.8-Flash-Next-FP8, `-tp 4 --enable-expert-parallel` on 4 GPUs (single node).
  The full serve command is in `trace.sh` next to this file. Any TP>1 model probably reproduces it.

## Symptom
1. FlashInfer logs, for every rank:
   `[MNNVL] FABRIC handles failed on this host; using POSIX file descriptors instead: ... CUDA error code=800 (CUDA_ERROR_NOT_PERMITTED)`
2. Then all 4 TP workers sit at about 100% CPU each, with the GPUs at 0% utilization,
   and nothing is logged for more than 20 minutes. EngineCore prints
   `No available shared memory broadcast block found in 60 seconds` every minute.
3. Python stacks, captured with `faulthandler` on SIGUSR1 (identical on all 4 ranks):
   ```
   torch/distributed/_symmetric_memory/__init__.py:2263  rendezvous
   vllm/distributed/device_communicators/custom_all_reduce.py:377  _init_mnnvl_buffer
   vllm/distributed/device_communicators/custom_all_reduce.py:350  __init__
   vllm/distributed/device_communicators/cuda_communicator.py:160  __init__
   vllm/distributed/parallel_state.py:1548  init_model_parallel_group
   vllm/distributed/parallel_state.py:2227  initialize_model_parallel
   vllm/v1/worker/gpu_worker.py  init_device
   ```
4. Another run got past this point (non-deterministic?), loaded the weights, and then hung
   later, right after FlashInfer's MNNVL warnings. Its stack was not captured.
5. Workaround that starts in about 2.5 minutes: `--disable-custom-all-reduce` plus
   `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`.

Logs: `base.hang1/serve.log` (first hang, after weight load) and `base.hang2/serve.log`
(stack dump above), in this directory.

## What we already know
- `CustomAllreduce.__init__` calls `self._init_mnnvl_buffer(...)` unconditionally for
  same-node groups. It does `torch_symm_mem.empty` + `torch_symm_mem.rendezvous` inside a
  `try`, but a hang is not an exception, so the `try` does not help.
- #59208 (348b4bf896, "Guard cross-node symmetric-memory rendezvous") added a
  `get_symmetric_memory_topology` guard, but only for **cross-node** groups. Single-node
  groups on a host without FABRIC support are not covered.
- #53253 gated cross-node MNNVL custom all-reduce by group capability. That is also cross-node only.

## Questions to answer
1. Where exactly does torch's CUDA symmetric-memory backend spin when the FABRIC handle
   export fails? Does it fall back to POSIX FDs, and does that exchange deadlock or
   busy-wait, for example on a store key that is never set? Is this a torch bug worth
   reporting upstream?
2. Why is it non-deterministic: sometimes it hangs at `_init_mnnvl_buffer`, sometimes later?
   Which later call is the second hang (FlashInfer MNNVL workspace? another rendezvous)?
3. What is the right vLLM-side guard? Options:
   (a) probe FABRIC/multicast support (for example a `cuMemCreate` with
       `CU_MEM_HANDLE_TYPE_FABRIC`, or reuse `get_symmetric_memory_topology` /
       FlashInfer's probe) and skip `_init_mnnvl_buffer` when it is unsupported;
   (b) an env switch to disable the MNNVL buffer;
   (c) a timeout on the rendezvous.
   Prefer (a). Same node with NVLink does not imply that FABRIC handles are usable.
4. Check whether `torch_symm_mem.set_backend("NCCL")` or `TORCH_SYMMMEM=...` avoids the hang,
   and whether the MNNVL buffer is even used on a single node without multicast
   (`handle.multicast_ptr == 0` makes the function return early anyway, so the rendezvous
   may be pure risk with no benefit in this case).
5. Write a minimal standalone repro (2 GPUs, torch only, no vLLM) if possible.

## Deliverable
A short write-up: root cause (torch vs vLLM vs environment), evidence (stacks, repro), and
a proposed vLLM patch with a unit test in the style of `tests/distributed/test_custom_all_reduce.py`.
Do not push anything or open any issue or PR. The owner decides.
