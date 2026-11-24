# B300 node: startup instructions

You are the **B300 worker** (role `b300`) in a three-node vLLM kernel-optimization round:

| Node | Role | Owns in this branch |
|---|---|---|
| 8× B200 | **manager**: assigns tasks, reviews results, writes and tests the PR code | `TASKS.md`, `TASKS_B300.md`, `STATUS_MANAGER.md`, `results/manager/` |
| 8× B200 | `worker` | `STATUS_WORKER.md`, `results/worker/` |
| 8× B300 (this node) | `b300` | `STATUS_B300.md`, `results/b300/` |

Your goal is to find B300-specific (SM103) performance opportunities in vLLM, measure them cleanly, and report them here. The manager decides what becomes a PR, and the human owner submits it.

## 0. Hard rules (read before anything else)

- **This fork is public.** Never commit tokens, emails, hostnames, IPs, user names or file paths that reveal them. Never commit unreviewed vLLM code patches. Describe a change in words plus at most a ~15-line snippet.
- **Never push vLLM code anywhere.** Never open, comment on or review PRs or issues on `vllm-project/vllm` or any other repo. The human owner does all upstream submission.
- **Push only to the `kopt-comms` branch, and edit only your own files** (`STATUS_B300.md`, `results/b300/`): `git pull --rebase origin kopt-comms && git push origin HEAD:kopt-comms`.
- **Commit identity for this channel:**
  ```bash
  git -C ~/kopt-comms config user.name kopt-b300
  git -C ~/kopt-comms config user.email kopt-b300@users.noreply.github.com
  ```
- **Commit messages** start with `b300: <task id> ...`, for example `b300: B1 done — env + capability survey`.
- **Poll** `git fetch origin kopt-comms` about every 10 minutes while you are waiting for a task. Not faster.
- **Report failures right away.** On an error, a hang or a result that looks wrong, push a short status update before retrying.

## 1. Get the channel

```bash
git clone -b kopt-comms --single-branch https://github.com/Johnsonms/vllm.git ~/kopt-comms
cd ~/kopt-comms && cat README.md TASKS_B300.md
```

Push access to `Johnsonms/vllm` is set up by the human owner (`gh auth login` as the owner's account, or a deploy credential). If `git push` fails, write that in your session and wait. Do not work around it.

## 2. Environment

Use the round's current base commit so numbers are comparable with the B200 nodes. The base is vLLM main `e66bede56b`, which has no C++ changes since the precompiled wheel.

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm-kopt
cd ~/vllm-kopt && git checkout e66bede56b
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in cupti-python ninja
vllm download-kernels             # FlashInfer cubins; without them FlashInfer kernels are disabled

cat > ~/kopt-env.sh <<'EOF'
source $HOME/vllm-kopt/.venv/bin/activate
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda
export HF_HOME=<your HF cache dir>       # keep any HF token out of the channel
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
EOF
```

For each A/B arm, use a separate git worktree of `~/vllm-kopt` and symlink the built artifacts (the `.so` files and `vllm/vllm_flash_attn/*`) from the main checkout. Alternatively, check out the arm's commit in one tree between runs.

## 3. Measurement rules (lessons the other two nodes paid for)

1. **Idle node.** Check `nvidia-smi` before every timed run. Run nothing else on the node during an A/B, not even on other GPUs: Qwen3.5/3.6 MoE runs with compilation off and breakable CUDA graphs, so it is host-sensitive. One contended run produced a fake 6% gain.
2. **Alternate arms** A, B, A, B, and report per-run numbers as well as the mean.
3. **Disable the compile cache for serving A/Bs:** `export VLLM_DISABLE_COMPILE_CACHE=1`.
   - The torch.compile AOT cache key does not include table files such as `skinny_gemm_plans.py`, so the second arm can silently load the first arm's graph.
   - Check that `grep -c 'Directly load AOT' serve.log` is 0 in every arm.
4. **cwd matters.** `python -m vllm...` puts the cwd first on `sys.path`. Launch from inside the tree you are timing, and print `vllm.__file__` (or the module you changed) into the log.
5. **`vllm bench forward` forces `CompilationMode.NONE`.** It is fine for kernel-time profiles, but it cannot evaluate compile-pass changes. Use `vllm serve` + `vllm bench serve` for those.
6. **Kernel timing:** use `flashinfer.testing.bench_gpu_time_with_cupti(..., use_cuda_graph=True, cold_l2_cache=True, input_args=...)`. Pass tensors through `input_args`, or cold-L2 is silently off.
7. **Shared experts run on an aux CUDA stream at decode** (`runner/shared_experts.py`, `MULTI_STREAM_OVERLAPPED` for tokens ≤ `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`). Speeding up shared-expert kernels usually shows zero E2E gain at decode. Check the critical path before spending GPU time.
8. **HF downloads:** pass one `--include` pattern per call, or download files by name. A multi-pattern `--include` silently skipped the tokenizer files once. Check with `du -shL <snapshot>`, because the cache keeps file contents in `blobs/`.
9. **Pitfalls:**
   - `torchrun` needs `--master-addr=127.0.0.1`.
   - `pkill -f` and `pgrep -f` match your own shell; use the `[x]yz` bracket trick.
   - Never pipe a long-running loop into `head`: SIGPIPE kills it.
   - Gated models (meta-llama, gemma) are not available; use ungated mirrors or Qwen/zai-org/deepseek-ai models.

## 4. Context: what the round has found so far (on B200, SM100)

- **The pattern that lands fastest:** B200-measured skinny-GEMM plan tables for BF16 decode projections at 1–8 tokens.
  - Selection: sweep, keep entries ≥ 10% over cuBLAS, then a 4-seed CUPTI verify.
  - Validation: E2E serving A/B with GSM8K.
  - #69529 (Qwen4Exp SM100 TP2/TP8) merged within a day.
  - #69736 (open) adds the **generic SM100** table for Qwen3.5/3.6 MoE: GDN `in_proj_ba`, router, shared-expert gate, LM head. It gave −5.9% TPOT at c4 on Qwen3.6-35B-A3B-FP8.
- **It registers `(10, 0)` only. B300 is `(10, 3)`, so on B300 all of those projections still run cuBLAS.**
- **SM103 (B300) coverage on main:**
  - `QWEN4_EXP_SM103_GEMM_PLANS` (`vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`) covers TP1 and TP4 only, with no TP2 or TP8.
  - The generic `SKINNY_GEMM_PLANS_BY_CAPABILITY` has no `(10, 3)` entry.
  - The AR+RMSNorm fusion cap (`FI_ALLREDUCE_FUSION_MAX_SIZE_MB` in `vllm/config/compilation.py`) is 4 MB at SM103 TP8. On SM100, raising TP8 from 1 to 16 MB gave −10% TPOT at c128 (#69543).
- **Tools:** the sweep tool is `results/manager/tools/sweep_skinny_generic.py SHAPES.json OUT.jsonl M...`. `results/worker/W11/` is a complete worked example of sweep, verify and report; follow its format.

## 5. Start here

Do **B1** in `TASKS_B300.md` first, then wait for the manager's review before starting B2.
