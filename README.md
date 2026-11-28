# kopt-comms — manager / worker channel

This orphan branch is the channel between two Claude Code sessions working on the
vLLM kernel-optimization round. It contains no vLLM source.

- **Manager**: the first B200 node. Owns `TASKS.md` and `STATUS_MANAGER.md`. Assigns work and merges results.
- **Worker**: the second node (B200). Owns `STATUS_WORKER.md` and `results/worker/`. Runs the tasks in `TASKS.md`.
- **B300** (`b300`): the third node (8x B300, SM103). Owns `STATUS_B300.md` and `results/b300/`. Runs the tasks in `TASKS_B300.md`. Start with `setup/B300_START.md`.
- **Hopper** (`hopper`): an SM90 node. Owns `STATUS_HOPPER.md` and `results/hopper/`. Runs the tasks in `TASKS_HOPPER.md`. Start with `setup/HOPPER_START.md`.
- **Hopper 2** (`hopper2`): a second SM90 node. Owns `STATUS_HOPPER2.md` and `results/hopper2/`. Runs the tasks in `TASKS_HOPPER2.md`. Start with `setup/HOPPER2_START.md`.

**This fork is public.** Never commit tokens, emails, hostnames/IPs, or unreviewed code patches here.

## Get the channel

```bash
git clone -b kopt-comms --single-branch https://github.com/Johnsonms/vllm.git ~/kopt-comms
```

The worker needs push access to `Johnsonms/vllm`; the human owner sets that up.

## Rules (both sides)

1. **Edit only your own files.** The manager edits `TASKS.md`, `TASKS_B300.md`, `TASKS_HOPPER.md`, `TASKS_HOPPER2.md`, `STATUS_MANAGER.md` and `results/manager/`; the worker edits `STATUS_WORKER.md` and `results/worker/`; the B300 node edits `STATUS_B300.md` and `results/b300/`; the Hopper node edits `STATUS_HOPPER.md` and `results/hopper/`; the second Hopper node edits `STATUS_HOPPER2.md` and `results/hopper2/`. This keeps rebases conflict-free.
2. **Push to this branch only:** `git pull --rebase origin kopt-comms && git push origin HEAD:kopt-comms`.
3. **Never push vLLM code anywhere.** Never open, comment on, or review PRs or issues on `vllm-project/vllm` or any other repo. The human owner does all upstream submission.
4. **Commit messages** start with the role and task ID, for example `worker: W1 done — kernel crossover table` or `manager: assign W2`.
5. **Poll** with `git fetch origin kopt-comms` every ~10 min while idle. Do not poll faster.
6. **Measure on an idle node.** Before every timed run, check `nvidia-smi` and make sure no other job uses the GPUs. Alternate A/B arms (A, B, A, B). Report per-run numbers as well as the mean.
7. **Report failures right away.** On an error, a hang, or a result that looks wrong, push a status update before retrying.

## Task lifecycle

`TASKS.md` lists tasks as `W<n>`. A task moves through: `assigned` (manager) → `ack` (worker, in `STATUS_WORKER.md`) → `running` → `done` or `blocked` (worker).
The manager reviews the result and either closes the task or replies in `TASKS.md` with follow-ups.

## Layout

```
README.md            protocol (this file)
TASKS.md             manager → worker: task specs, priorities
STATUS_MANAGER.md    manager's own progress / context
STATUS_WORKER.md     worker → manager: ack, progress, results summary, questions
setup/SETUP.md       how to reproduce the manager's environment
results/worker/W<n>/ raw logs/tables from the worker (keep each file < 5 MB)
results/manager/     manager-side reference numbers
```
