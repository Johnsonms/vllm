# W19 — TP>1 startup hang in torch symm-mem rendezvous: interim (2026-11-28 06:36Z)

## Short version
- **"No FABRIC" is not the cause.** My node is the same case (no IMEX channel, same FlashInfer warning, torch falls back to POSIX fds) and starts every time, including **your exact SHA 14e9c30d86** (healthy in 260 s).
- **So a guard that skips the MNNVL buffer when FABRIC is missing would be wrong:** without FABRIC, the rendezvous here still returns a nonzero multicast pointer, so the buffer is in use on hosts like mine.
- I can reproduce **one** rendezvous hang (a stale socket file). It does not fully match your stacks, so I need 4 facts from your node (last section) to say whether it is yours.

## 1. Contrast: my node (8x B200, single node, in a container)
- `/dev/nvidia-caps-imex-channels` does not exist. Only `nvidia-cap1` and `nvidia-cap2` under `/dev/nvidia-caps`.
- torch 2.14.0+cu132, FlashInfer 0.7.4, NCCL 2.30.7, driver 580.159.04.
- With `TORCH_CPP_LOG_LEVEL=INFO`, every rank logs: `PeerToPeerAccess.cpp:125 status 800 Could not allocate memory with FABRIC handle, falling back to fd handle exchange`, then `has_fabric_support 0`, then `using posix fd to import symmetric memory handles`. The rendezvous takes about 1 s and `multicast_ptr` is nonzero on all 4 ranks.
- TEP4 Flash-Next starts: 20+ starts in W16 / W17 (main 0fbba2aa71, 361f362033) and one start of 14e9c30d86 today. The FlashInfer `[MNNVL] FABRIC handles failed ... using POSIX file descriptors` warning appears on all 4 ranks, **minutes after** the communicators are built (the workspace is created lazily at model load), followed by `Initialized FlashInfer Allreduce norm fusion workspace with backend=mnnvl`.
- Order inside `CudaCommunicator.__init__` at 14e9c30d86: `SymmMemCommunicator` (first torch rendezvous, `symm_mem.py:110`), then `FlashInferAllReduce`, then `CustomAllreduce` (second torch rendezvous, `_init_mnnvl_buffer`).
- **Idle healthy workers use 115-130 % CPU each here.** So "100 % CPU" alone does not prove a spin inside the rendezvous.

## 2. What the torch source says (v2.14.0, `CUDASymmetricMemory.cu`, `CUDASymmetricMemoryUtils.cpp`)
- The allocator picks FABRIC or POSIX fd once per process from a FABRIC probe. With POSIX fds, each rank binds an AF_UNIX datagram socket at `$TMPDIR/symm_mem-<pid>` (default `/tmp`), exports its block, exchanges metadata through the group store, then passes fds around a ring with blocking `sendmsg` / `recvmsg`.
- Multicast: rank 0 creates the object, broadcasts the fd, every rank imports it, adds its device and binds memory; then a success-flag all-gather through the store.
- **No step has a timeout of its own**, and each step assumes every rank reaches it. If one rank throws before a step, the others block in that step: in the store get, in `recvmsg`, or (per a comment in the source) in `cuMulticastBindMem`, which waits until every device was added.
- The socket file is removed only by the channel's destructor. A process killed inside the rendezvous leaves its file behind.

## 3. Reproduced: stale socket file -> one rank raises, the others hang
`repro.py` (torch only, in this folder): `CUDA_VISIBLE_DEVICES=a,b,c,d python repro.py 4` is the normal case; `python repro.py 4 1` makes rank 1 create `$TMPDIR/symm_mem-<own pid>` first.
- Rank 1: `RuntimeError: Failed to bind socket: Address already in use` after 0.5 s. With the error swallowed (as vLLM does) it goes on to the next collective and blocks there.
- Ranks 0, 2, 3: blocked in `rendezvous` (`__init__.py:2263`), CPU idle.
- Killing the hung job leaves the 3 other socket files behind, so the next job whose pid matches one of them fails the same way. In a container with a small pid space this repeats itself.
- Same in real vLLM (TEP4, stale files for one pid in four, private TMPDIR): one worker logs `SymmMemCommunicator: symmetric memory initialization failed: Failed to bind socket: Address already in use`; the other 3 sit in `rendezvous` called from `symm_mem.py:110`; never healthy.
- **Why this may not be yours:** it hits the first rendezvous (`symm_mem.py:110`), and 3 of 4 ranks are in it. You report 4 of 4 in the second one (`custom_all_reduce.py:377`).

Also tested, no hang: a second 4-rank rendezvous on the same GPUs while another process group holds its symmetric memory, and one on an overlapping GPU set.

## 4. What I need from the manager node to decide
1. `ls $TMPDIR/symm_mem-* /tmp/symm_mem-* 2>/dev/null | wc -l` before a start, and whether the hung run's log has `Failed to bind socket` anywhere (also at debug level: `MNNVL AG/RS initialization failed`).
2. FlashInfer version, and the lines of `base.hang2/serve.log` between `Using [...] all-reduce backends` and the hang. On my node the FlashInfer MNNVL warning cannot come before `_init_mnnvl_buffer` (lazy workspace). If it does on yours, something creates the FlashInfer workspace earlier there.
3. Native stacks of one hung worker: `py-spy dump --native --pid <pid>` (or `gdb -p <pid> -batch -ex 'thread apply all bt'`). This tells store get vs `recvmsg` vs `cuMulticastBindMem`. The Python stack cannot.
4. One start with `TORCH_CPP_LOG_LEVEL=INFO`: the last symm-mem line per rank shows how far each rank got, and whether all ranks chose the same handle type.
5. Optional: `repro.py 4` on the same 4 GPUs. If it hangs there, it is torch plus the host, and vLLM is not involved.

## 5. Guard: direction (final version after the facts above)
- Not "FABRIC missing -> skip" (section 1).
- Make the decision collective before entering the rendezvous: each rank checks its local preconditions (symm-mem importable, device multicast support, its socket path free), the group all-gathers the result, and all ranks enter or all skip. That removes the "one rank raises, the others wait forever" class for both `SymmMemCommunicator` and `_init_mnnvl_buffer`.
- For the stale file itself: a rank may delete `$TMPDIR/symm_mem-<own pid>` before the rendezvous, because a file with its own pid that it did not create is stale by definition. This is really a torch issue (bind without unlink, no cleanup on kill, no timeout); worth reporting upstream once confirmed.
- Unit-test plan and the remaining brief questions (timeouts, `TORCH_SYMMMEM=NCCL`) follow in the final report.
