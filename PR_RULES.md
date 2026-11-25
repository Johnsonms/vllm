# vLLM PR rules for this team (manager, B200 worker, B300)

These are the rules the manager has used for #69529, #69736, #69543, #69575 and #69778. Follow all of them for any PR, push or GitHub comment.

## 1. Approval: nothing goes out without the owner's OK
- **Never push to GitHub, open a PR or issue, or post a comment or review without the owner's explicit OK for that specific action.** An OK for one action does not carry over to the next.
- Before asking the owner, send the PR to the manager for review via `results/<node>/PRREVIEW/SUMMARY.md`. Include the title, base commit, `git diff --stat`, the evidence (section 4) and the duplicate check (section 2).
- Local commits, local tests and benchmarks are always fine.

## 2. Before writing code: check for duplicates
- Search open **and** merged PRs: `gh pr list -R vllm-project/vllm --state all --search "<keywords>"`. Try 3–4 phrasings (kernel name, op name, model name, file name).
- If an open PR already does the same thing, do not open a competing one. Report it to the manager instead. Examples from this round: the sampler top-k fast path (#64116), and Gemma RMSNorm + block-FP8 quant (#65497, #36016, #45770).
- Do not overlap work in flight on this team. For example, the bit-identical SiLU-and-mul + block-FP8 producer is already in #69778.
- If a related PR exists but does something different, list it in the body (section 5) and say how the two interact.

## 3. Commits
- **Identity.** Author **and** committer must be the owner's GitHub identity, the same as on #69736 and #69778, with a `Signed-off-by:` line. Set them repo-locally (`git config user.name` / `user.email`); do not rely on the machine default.
- **No `Co-authored-by:` line.** Remove it if a tool adds one.
- **Check after every rebase, cherry-pick or amend:**
  ```
  git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD
  ```
  A plain rebase rewrites the committer to the machine default. That leaked a cluster hostname once (#69773), and the old SHA stays visible in the PR timeline forever.
- **Commit message.** The subject is the PR title. The body says what changed and why, in plain sentences, and ends with Signed-off-by.
- **One topic per PR.** vLLM squash-merges, so the PR title becomes the commit subject on main. Several commits inside one PR are fine (for example, wiring plus a kernel fix); you don't need to squash them yourself.
- Run `pre-commit run --files <changed files>` (for example via `uvx pre-commit`) before pushing. It must pass with no files modified.

## 4. Evidence required before asking to open
1. **Unit tests.** Add a test that fails without the change and passes with it. Report the command, the counts, and the result on both the PR and the base.
2. **Correctness and accuracy.**
   - Prefer bit-identical output to the base path: `torch.equal` in the test, and greedy tokens identical on 32 single prompts and 256 batched prompts.
   - If the output is not bit-identical, run GSM8K at least 8 runs per arm with per-question dumps. Report the base-vs-base flip floor, and expect to be asked why.
   - #69778's first version lost about 0.4 GSM8K points this way, and it took a kernel rewrite to close that.
3. **Kernel microbench.** CUDA graphs or CUPTI, a cold L2 where it matters, several seeds and interleaved rounds. Report the speedup range and any shapes that got slower.
4. **End-to-end, paired A/B.**
   - Run both servers at once on two GPUs, alternating short bursts (5 × c prompts, order flipped each round, at least 4–6 rounds).
   - Then swap the GPUs and repeat. Report the mean of the two passes, and how many pairs were faster.
   - Sequential A/Bs are not accepted: they faked 6–8% here.
   - Set `PYTHONPATH=<tree>` per server, because otherwise `vllm serve` imports the venv's editable install. Log the loaded file path and HEAD in each server log.
   - Set `VLLM_DISABLE_COMPILE_CACHE=1`, and confirm "Directly load AOT" appears 0 times in every server log.
   - Report median TPOT at c = 1, 2, 4, 8 at least, and higher c if the change could regress there.
5. **Say what was not measured** (other GPUs, other models, SM90 and so on), and say it if anything was built in an unusual way (for example, a JIT-built `.cu`).

## 5. PR title and body
- **Title:** a `[Perf]` / `[Kernel]` / `[Bugfix]` / `[Model]` tag, then one plain sentence saying what it does and where. For example: `[Perf] Run small-M block-FP8 GEMMs on FlashInfer's TRT-LLM kernel on SM100`.
- **Body sections, in this order:**
  1. **Overview:** what was slow or wrong, why, what the PR changes, and what is not affected.
  2. **Results:** the end-to-end table (method stated in bullets), the kernel-level table, and accuracy.
  3. **Testing:** commands and counts, and pre-commit.
  4. **Duplicates and overlap / Related PRs:** each related PR and how they interact.
  5. **AI assistance:** one sentence saying Claude Code was used and what for, plus "The submitter reviewed every changed line." End with the line `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Style:** plain and exact. Use numbers with their method, and no marketing words. Report negative or flat results too; do not hide a slower case. Write each number once, and check every number against the raw logs before publishing.
- Keep the draft locally (for example `prs/<name>_body.md`) and keep the old version when you revise.

## 6. After opening
- **Never request the `ready` or `verified` labels**, and never ask for CI to be triggered. The CI log says AI agents must not request them.
- Do not `@`-mention people unless the owner asks you to.
- Reply to reviewers only with the owner's OK. Draft the reply first and send it to the manager or the owner.
- When fixing review comments, push new commits (or a force-push only with the owner's OK). Re-check the commit identity (section 3), and update the body if numbers change.
- Small related tunings go into an existing PR rather than new ones if a reviewer asks for that. #69743, #69773 and #69750 were folded into #69736 this way.

## 7. This comms branch is public
- Never put tokens, email addresses, hostnames, IPs, or unreviewed code patches here. Describe code changes in words; share code through a branch only after the owner approves the push.
- Use the `kopt-<node>` commit identity for comms commits, not the owner's identity.
