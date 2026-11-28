# Second Hopper node: startup instructions (role `hopper2`)

You are the **second Hopper worker** (role `hopper2`, 8x H100, SM90). The first Hopper node (role `hopper`) keeps running its current end-to-end tests; you take over new Hopper work, and over time the remaining Hopper work moves to you.

Everything in `setup/HOPPER_START.md` applies to you (hard rules, environment, measurement rules) with these substitutions:
- **Your files:** `STATUS_HOPPER2.md` and `results/hopper2/`. Edit nothing else. Tasks for you are in `TASKS_HOPPER2.md`.
- **Commit identity:**
  ```bash
  git -C ~/kopt-comms config user.name kopt-hopper2
  git -C ~/kopt-comms config user.email kopt-hopper2@users.noreply.github.com
  ```
- **Commit messages** start with `hopper2: <task id> ...`.
- Read `STATUS_HOPPER.md` and `results/hopper/` to see what the first node found (H1 env, H2 MoE/routing, H3 SwiGLU, H5 permute). Reuse its scripts and conventions where useful; do not edit its files.
- Record your node's `nvidia-smi topo -m` (NUMA affinity) and check every GPU's idle temperature and clocks first: the first node has one outlier GPU (idle ~71 °C, ~8% slow). Exclude any such GPU from paired E2E and report it.

## Start
```bash
git clone -b kopt-comms --single-branch https://github.com/Johnsonms/vllm.git ~/kopt-comms
cd ~/kopt-comms && cat README.md setup/HOPPER_START.md setup/HOPPER2_START.md TASKS_HOPPER2.md
```
Then do the tasks in `TASKS_HOPPER2.md` in order.
