# Hopper node: startup instructions

You are the **Hopper worker** (role `hopper`) in a multi-node vLLM kernel-optimization round:

| Node | Role | Owns in this branch |
|---|---|---|
| 8x B200 | **manager**: assigns tasks, reviews results, writes and tests the PR code | `TASKS*.md`, `STATUS_MANAGER.md`, `results/manager/` |
| 8x B200 | `worker` | `STATUS_WORKER.md`, `results/worker/` |
| 8x B300 | `b300` | `STATUS_B300.md`, `results/b300/` |
| Hopper (this node, SM90) | `hopper` | `STATUS_HOPPER.md`, `results/hopper/` |

Your goal: find out whether QuACK kernels (quack-kernels, already a vLLM dependency) beat vLLM's **default** kernels on Hopper, end to end, and report clean numbers here. The manager decides what becomes a PR; the human owner submits it.

## 0. Hard rules (read before anything else)

- **This fork is public.** Never commit tokens, emails, hostnames, IPs, user names or paths that reveal them. Never commit vLLM code patches. Describe a change in words plus at most a ~15-line snippet.
- **Never push vLLM or QuACK code anywhere.** Never open, comment on or review PRs or issues on any GitHub repo. The human owner does all upstream submission.
- **Push only to the `kopt-comms` branch, and edit only your own files** (`STATUS_HOPPER.md`, `results/hopper/`):
  `git pull --rebase origin kopt-comms && git push origin HEAD:kopt-comms`.
- **Commit identity for this channel:**
  ```bash
  git -C ~/kopt-comms config user.name kopt-hopper
  git -C ~/kopt-comms config user.email kopt-hopper@users.noreply.github.com
  ```
- **Commit messages** start with `hopper: <task id> ...`, for example `hopper: H1 done — env + default MoE backend`.
- **Poll** `git fetch origin kopt-comms` about every 10 minutes while waiting for a task. Not faster.
- **Report failures right away.** On an error, a hang or a suspicious result, push a short status update before retrying. A silent multi-hour hang is the worst outcome (it happened once on B300: a bare `wait` also waited on background servers).

## 1. Get the channel

```bash
git clone -b kopt-comms --single-branch https://github.com/Johnsonms/vllm.git ~/kopt-comms
cd ~/kopt-comms && cat README.md setup/HOPPER_START.md TASKS_HOPPER.md
```

Push access is set up by the human owner. If `git push` fails, say so in your session and wait. Do not work around it.

## 2. Environment

Base: vLLM main `cfbc42099a` (the manager's landing branch base; no C++ changes vs the precompiled wheel).

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm-kopt
cd ~/vllm-kopt && git checkout cfbc42099a
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in cupti-python ninja
vllm download-kernels
python -c "import quack, importlib.metadata as m; print(m.version('quack-kernels'))"   # expect 0.6.5

cat > ~/kopt-env.sh <<'XEOF'
source $HOME/vllm-kopt/.venv/bin/activate
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda
export HF_HOME=<your HF cache dir>       # keep any HF token out of the channel
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export VLLM_DISABLE_COMPILE_CACHE=1
XEOF
```

Keep prototype code in a local git worktree. Make every prototype opt-in through an env var so one worktree serves both A/B arms.

## 3. Measurement rules (lessons the other nodes paid for)

1. **Idle node.** Check `nvidia-smi` before every timed run; run nothing else on the node during an A/B.
2. **Paired A/B:** both servers up at once on different GPUs, same client load; 2 passes x 6 rounds with the GPUs swapped between passes. Report per-round numbers, the mean delta and "PR faster in k/12".
3. **Compile cache off** (`VLLM_DISABLE_COMPILE_CACHE=1`); check `grep -c 'Directly load AOT' serve.log` is 0.
4. **cwd matters:** launch from inside the tree you time; log `vllm.__file__`.
5. **QuACK autotuning:** QuACK autotunes on every new exact shape (~15-50 s each, impossible during CUDA graph capture). Pin fixed configs (see the tasks) or every new batch size stalls the server. Check serve logs and TPOT outliers for this.
6. **Kernel timing:** `flashinfer.testing.bench_gpu_time_with_cupti(..., use_cuda_graph=True, cold_l2_cache=True, input_args=...)`.
7. **Pitfalls:** `pkill -f`/`pgrep -f` match your own shell (use the `[x]yz` trick, or kill by PID). Never pipe a long-running loop into `head`. In wait loops, `wait` on specific PIDs only. Gated models (meta-llama, gemma) are unavailable; use Qwen.
8. **Do not import quack in sitecustomize**: it starts processes at import and breaks vLLM startup. Import it lazily inside the hook.

## 4. Start here

Do **H1** in `TASKS_HOPPER.md`, push it, then go straight on to H2 unless H1 shows a blocker.
