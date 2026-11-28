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

## 6. Reply to the manager's facts (d62dfe1), 2026-11-28 06:43Z

**Your fact 5 is the strongest lead: NCCL splitting one rank off points at that GPU, not at FABRIC.**
- On my node, the same 4-rank group on GPUs 2-5 with `NCCL_DEBUG=INFO` reports **`nRanks 4 nNodes 1 localRanks 4`** on all 4 ranks, with the same fabric line as yours (`fabric UUID 0.0 cliqueId 0x0 state 3 healthMask 0x80`). I have never seen `nNodes 2` for TP4 here.
- NCCL does not get `nNodes` from the hostname. In `init.cc` each rank computes its local ring from its own topology graph and reports the ring's first rank; `nNodes` is the number of distinct first ranks. The host hash is hostname + boot id (+ communicator hash), which is the same for all processes of one container. So `nNodes 2` with `localRanks 1` on rank 2 means: **in the topology that NCCL built, physical GPU 4 is not connected to the other three as a local peer** (and the other three agree, with `localRanks 3`).
- That fits the hang. torch's multicast setup has every rank add its device to one multicast object and then call `cuMulticastBindMem`. The torch source says this call waits until all devices have joined. If one GPU cannot join the same NVLink multicast group, every rank stays inside the rendezvous, busy in the driver, with the GPU idle: 4 of 4 ranks in the rendezvous, at whichever rendezvous comes first (`symm_mem.py:110` in hang3, `custom_all_reduce.py:377` in hang2). **This is a hypothesis until the checks below.**
- Your fact 1 rules out my stale-socket case for these runs: the 8 files were created by the hung workers themselves, and no rank raised. (They will hurt later: remove `/tmp/symm_mem-*` once no job is running, or a later job with a recycled pid fails to bind.)

**Checks, cheapest first (all with `repro.py`, torch only):**
1. `CUDA_VISIBLE_DEVICES=2,3,4,5 python repro.py 4`. Expect a hang if the above is right.
2. Same GPUs with `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`. torch then skips the multicast phase and keeps the fd exchange. **If this finishes and check 1 hangs, the hang is in the multicast bind.**
3. Leave GPU 4 out: `CUDA_VISIBLE_DEVICES=2,3,5,6 python repro.py 4`. If this finishes, it is GPU 4. Then pairs: `4,5` and `2,3` with `python repro.py 2`.
4. `nvidia-smi nvlink -s -i 4` and `nvidia-smi topo -m`. Here every GPU has all links up at the same speed and every GPU pair shows `NV18`. A down link or a different `NV` count on GPU 4 would confirm it.
5. The `NCCL_DEBUG=INFO` lines of rank 2 from hang2 about topology or P2P (anything that says why it found no local peers).
6. Native stack of one hung rank, if ptrace works: the top frames should be inside `libcuda` under `cuMulticastBindMem`.

**Was hang1 on the same 4 GPUs?** It passed both init rendezvous and hung later. If it used the same GPUs, the GPU-4 hypothesis needs torch to have skipped multicast in that run; the `TORCH_CPP_LOG_LEVEL=INFO` start will show it (`Gracefully skipping multicast initialization`).

**Timeout test (my node):** 2 ranks, one with a stale socket, left alone for 400 s: both still blocked at 400 s (CPU idle), no error raised. So nothing in this path times out within the 300 s store default.

**Guard, if this is confirmed.** Neither FABRIC nor the per-device multicast attribute detects it: the device says it supports multicast, the group cannot form one. Options, in my order of preference:
1. Use NCCL's own answer. vLLM already has the NCCL communicator for the group; if it reports more than one node for a group that vLLM believes is single-node, treat the group as not multicast-capable and skip both symm-mem rendezvous (SymmMemCommunicator and the MNNVL buffer) on all ranks. This is collective by construction (all ranks see the same count).
2. An environment switch that turns off both rendezvous. Today this needs `--disable-custom-all-reduce` plus `VLLM_ALLREDUCE_USE_SYMM_MEM=0` plus the FlashInfer backend variable; one documented switch would do.
3. A watchdog around the rendezvous that aborts the worker with a clear message after N seconds. It cannot recover (the driver call is not interruptible), but it turns a silent hang into an error that names the cause.
- Upstream torch: the multicast bind has no timeout and no pre-check that the devices can form a group; worth an issue once reproduced with `repro.py` alone.

## 7. FINAL (2026-11-28 06:55Z): root cause, guard proposal, unit-test plan

### Root cause
- **Environment, exposed by a missing check in vLLM and a missing timeout in torch.** On the manager node physical GPU 4 has no active NVLink (manager: `nvidia-smi nvlink -s -i 4` shows 0 links up; `topo -m` shows `SYS` / `PHB` to every other GPU, where healthy pairs show `NV18`).
- A TP4 group on GPUs 2-5 therefore contains one GPU that cannot join an NVLink multicast group. torch's symm-mem rendezvous still tries: every rank imports the multicast object, adds its device and calls `cuMulticastBindMem`, which waits for all devices. No rank raises, so vLLM's `try / except RuntimeError` never fires. All ranks stay in the rendezvous.
- It is not FABRIC: my node has no IMEX channel either and runs the POSIX-fd path with working multicast (section 1).
- Status of the evidence: the NVLink state and the NCCL split are measured on the manager node; the contrast and the source reading are mine. **The native stack that should show `cuMulticastBindMem`, and `repro.py` with and without `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`, are still to come from the manager.** Until then "bind" is the most likely blocking call, not a proven one.

### Answers to the brief
1. *Where does torch spin, is it a torch bug?* There is no busy-wait loop in torch's rendezvous; the waits are blocking calls (store get, `recvmsg`, driver calls). Most likely call here: `cuMulticastBindMem`. Two things are worth an upstream torch issue once `repro.py` reproduces it alone: (a) no pre-check that the group's devices can form a multicast group, and no timeout on the bind; (b) the fd-exchange socket at `$TMPDIR/symm_mem-<pid>` is bound without removing a stale file and is not cleaned up when the process is killed (section 3).
2. *Why non-deterministic?* Not answered from my node. A link that is inactive only some of the time would explain a run that passed both init rendezvous (hang1) and later hung in FlashInfer's own MNNVL multicast setup, which does the same add-device and bind sequence. The manager's `TORCH_CPP_LOG_LEVEL=INFO` start plus the link state at that moment would settle it.
3. *Right guard?* Below. A FABRIC probe is the wrong signal. The device-level multicast attribute is also not enough: a GPU with no link up still reports it.
4. *`TORCH_SYMMMEM=NCCL` / is the buffer used on one node?* On a healthy single node the rendezvous returns a nonzero multicast pointer (measured here), so `_init_mnnvl_buffer` does not return early and the buffer is registered. `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` is the torch switch that skips the multicast phase; then `multicast_ptr == 0` and the function returns early. I did not test `TORCH_SYMMMEM=NCCL`.
5. *Standalone repro:* `repro.py` in this folder (torch only). It needs a GPU without NVLink to show the bind hang, so only the manager node can run that case; here it shows the normal case and the stale-socket case.

### An existing check that should have caught it, and did not
`CustomAllreduce.__init__` already calls `current_platform.is_fully_connected(physical_device_ids)` and returns early ("not supported on more than two PCIe-only GPUs") when it is false. hang2 reached `_init_mnnvl_buffer`, which is after that return. **So on the manager node `is_fully_connected` answered true for GPUs 2-5 although GPU 4 has no link up.** It asks NVML for the pairwise P2P status with the NVLink capability index; that evidently does not reflect the live link state. Manager: please confirm by calling it for `[2, 3, 4, 5]` on your node.

### Proposed guard (prose only; nothing written into the tree)
1. **One helper, "every rank of this single-node group has live NVLink".** Each rank asks NVML for its own GPU's links (the links with the P2P capability) and counts the active ones; the result "at least one active" is all-reduced with MIN over the group, like the existing `_all_ranks_true`. FlashInfer's `support_nvlink` does the same per-device count, so the method is already trusted in this code base. If NVML is unavailable, return "unknown" and keep today's behaviour.
2. **Use it in three places, on all ranks alike (the vote is collective, so no rank enters alone):**
   - `SymmMemCommunicator.__init__`: next to the existing multi-node topology check. If false: warn with the rank and GPU that has no link, communicator not available.
   - `CustomAllreduce.__init__`: combine it with `fully_connected`. If false: take the existing "disabled" return, which also skips `_init_mnnvl_buffer`.
   - FlashInfer backend selection (`_resolve_fi_ar_backend`): if false on a single node, do not pick `mnnvl`. I have not checked whether `trtllm` works across a GPU without NVLink; if it does not, the answer is no FlashInfer all-reduce for that group.
3. **Message.** The warning should say which physical GPU has no active NVLink and that the fast all-reduce paths are off because of it. The real fix is to repair the link or not to put that GPU in the group.
4. **Not proposed:** a FABRIC probe (wrong signal, section 1); a thread timeout around the rendezvous (the driver call cannot be interrupted, and peers would be left in an undefined collective state). A watchdog that aborts the worker with a clear error is acceptable as a second line of defence.

### Unit-test plan (style of `tests/distributed/test_custom_all_reduce.py`, all mocked, no GPU)
1. *Helper, local part:* fake NVML with 18 links active -> true; all links inactive -> false; link indices beyond the device's count raising InvalidArgument are skipped; NVML import failure -> "unknown".
2. *Helper, group part:* local true but the mocked all-reduce returns 0 -> false (pattern of `test_cross_node_mnnvl_gate_requires_support_on_every_rank`).
3. *`CustomAllreduce`:* same-node group, 4 ranks, `is_fully_connected` mocked true, helper false -> `disabled` stays true, `_init_mnnvl_buffer` and `torch_symm_mem.rendezvous` are never called (mock raises if called), one warning. Helper true -> `_init_mnnvl_buffer` called once.
4. *`SymmMemCommunicator`:* helper false -> `disabled`, `torch_symm_mem.empty` / `rendezvous` never called. Helper true -> rendezvous called once.
5. *FlashInfer backend:* single node, auto, helper false -> not `mnnvl`.
6. *Regression for healthy hosts:* helper true and FABRIC unavailable -> everything is attempted as today (this is my node's case and must keep working).
7. On real hardware: `repro.py` on a group with and without the bad GPU, before and after the patch (manager node only).

## 8. Knob script for the manager node (2026-11-28 07:20Z): `repro2.py`
Torch only, hard deadline, prints a line before and after every step, removes only its own socket files. On my node all four settings below finish in about 10 s (each rendezvous 0.06-0.15 s, multicast pointer nonzero; 0 with multicast disabled).

What vLLM does that `repro.py` did not: **two rendezvous on a sub-group ('tp:0'), the first 32 MiB (SymmMemCommunicator), the second 96 MiB (`_init_mnnvl_buffer`)**. Your INFO log shows "using posix fd" twice per rank before the hang, which fits "the first one passed, the second one blocks" or "the large one blocks".

Run on GPUs 2-5, in this order; stop at the first that hangs and tell me which line each rank printed last:

| # | command prefix (then `CUDA_VISIBLE_DEVICES=2,3,4,5 python repro2.py 4`) | tests |
|---|---|---|
| 1 | `SIZES_MB=32,96 GROUP=sub` | the vLLM sequence |
| 2 | `SIZES_MB=96` | size alone, one rendezvous |
| 3 | `SIZES_MB=32` | the first vLLM size alone |
| 4 | `SIZES_MB=1,1` | two rendezvous, small |
| 5 | `SIZES_MB=1,1 TORCH_SYMMMEM_IMPLICIT_POOL=0` | two rendezvous without the shared pool |
| 6 | `SIZES_MB=32,96 GROUP=sub TORCH_SYMM_MEM_DISABLE_MULTICAST=1` | must pass if the bind is the only problem |
| 7 | `SIZES_MB=32,96 GROUP=sub WARM_NCCL=1` | NCCL communicator created first, like vLLM |
| 8 | the first hanging setting on `CUDA_VISIBLE_DEVICES=2,3,5,6` | must pass without GPU 4 |
| 9 | the first hanging setting on `CUDA_VISIBLE_DEVICES=4,2,3,5` | GPU 4 as rank 0 (the multicast creator) |

If 2 hangs and 3-4 pass, it is size. If 4 hangs, it is the second rendezvous (5 then says whether the shared pool matters). If none hangs, the trigger is outside torch's rendezvous alone (order with NCCL or with custom all-reduce's CUDA IPC buffers) and I will extend the script.

A second Codex pass on the torch multicast code (create / add-device / bind order, granularity, what the implicit pool does on a second rendezvous) is running; I will add its answers per candidate.

**`is_fully_connected` (your note):** agreed, and it goes into the guard. `nvmlDeviceGetP2PStatus` with the NVLink index answers "is P2P over NVLink permitted for this pair", not "is a link up". Fix direction: after the pairwise status, also require at least one active link per device (`nvmlDeviceGetNvLinkState` over the P2P-capable links), which is the same per-device count as the helper in section 7. With that, `CustomAllreduce` takes its existing "not fully connected" return on your node, and the helper only has to be added to `SymmMemCommunicator` and the FlashInfer backend choice. NCCL's node count is not usable directly: NCCL prints it but has no API for it.
