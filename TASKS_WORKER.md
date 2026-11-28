
### Manager review: W16 final (`6ea46f19e7`)
- Accepted: gate-only set (11 entries for (1, 2560)). c2–c16 −0.7/−2.3/−3.0/−0.9% clear of spread, c32 flat, c1 not claimed. GSM8K 8+8 0.9676 vs 0.9677, no systematic flips.
- Greedy: 2/32 single prompts differ (base-vs-base 0). Expected for a kernel switch (skinny vs cuBLAS accumulate differently), but the PR body must say so plainly, with the GSM8K per-question flip analysis as the accuracy evidence.
- Before a PRREVIEW (W17):
  1. Rebase onto current main. Add the (1, 2560) shape to the existing table unit test if one covers the SM100 Flash-Next table, and run it.
  2. Author and committer = owner identity, Signed-off-by, no Co-authored-by (trailer question pending with the owner).
  3. On the final SHA: re-run greedy single 32/32 and a short ABAB (3 starts, c1–c32), plus GSM8K 8+8. The SHA that opens is the SHA that was measured.
  4. Draft the title and body in `results/worker/W17/PRREVIEW.md`: what, why, a results table, "not measured" (TP8, SM103, MTP, #69907 base), AI disclosure and the Claude Code footer. Text only, no code patch on comms.
- #69907 is still open, with no review. If it merges before we open, the A/B must be repeated against it.
- Do not push the branch or open anything. Who opens is pending the owner.

### Owner decisions (2026-11-26)
- **The worker opens W15** (gate entries) itself once the manager posts `APPROVED TO OPEN: W15 @ <sha>`. Same process as B300's (TASKS_B300.md, 2d5cc9bc01): post the PRREVIEW (summary, title, body, diffstat, SHA) on comms, wait for approval, then open with the owner identity. Do not open a draft.
- **W15 may open before skinny.** It does not depend on #69949.
- **Commit trailer, new vLLM commits:** `Assisted-by: Claude Code` placed directly above `Signed-off-by:`. Still no `Co-authored-by`. Author and committer = owner identity; check `git log --format='%an <%ae> | %cn <%ce>'` before pushing.
- Adding the trailer by amending only the message keeps the tree identical. Post both the tree hash (`git rev-parse <sha>^{tree}`) and the new SHA; if the tree matches the measured commit, the measurements carry over.

### W18: #69778 sampled-eval check (maintainer question). Runs alongside W17, on GPUs W17 does not use.
Context: gau-nernst asked whether the bitwise-parity commit is needed. Fused kernels such as DeepGEMM and FlashInfer keep FP32 throughout and multiply by the reciprocal. They asked whether our GSM8K was temp 0 (yes), and suggested a sampled eval, since the original numerics may happen to favour greedy decoding.
- **Arms** (Qwen3.6-35B-A3B-FP8, TP1, same serve flags as W12):
  - main `fb159244e6`, the PR base;
  - v1 `c5ef00f633`, fp32 product;
  - v2 `152bca1e65`, bit-identical.
  - Use the same builds and the JIT-extension method as W12.
- **Eval:** `tests/evals/gsm8k/gsm8k_eval.py --use-chat-completions --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --max-concurrency 64`, with `--temperature 0.6` (top_p 0.95 if supported). 8 runs per arm with distinct seeds, if the script supports seeds. Halves on swapped GPUs.
- **Report:**
  - mean, sd and Welch t for v1 vs main and for v2 vs main;
  - the per-question analysis from W12 (questions that differ by ≥ 4 of 8);
  - "Directly load AOT" count 0;
  - the setup in a form we can paste into the PR thread (no hostnames).
- Priority: below W17 (W15 is next to open). Do not post anything on GitHub.

### Ping (2026-11-26 08:50Z)
No worker post since `6ea46f19e7` (05:26Z). Please acknowledge W17, W18 and the owner decisions (`0f107fd460`, `32909e884c`, `e0cb04f6c7`), and post a one-line status: running, blocked or idle.

### W18 moved to B300 as B9 (2026-11-26 13:30Z). If you already have W18 results, post them anyway. W17 stays with you.

### Manager review of the W17 PRREVIEW (`ee6b126233` + `c0ca4a682b`), 2026-11-27
The measurements and the body are good: the gain is shown at c4/8/16 with 3 of 3 starts, greedy matches 32/32, GSM8K is unchanged, and the drop from W16 is reported honestly. **Not approved yet, for one blocker:**

1. **Diffstat.** It lists 172 files (+629/−856), but the text says 2 files (the table plus one test). That diffstat is against some other base, not the parent of the commit. Post the output of these three commands:
   - `git rev-list --count 361f362033..383e366bcb` (expect 1)
   - `git show --stat --format= 383e366bcb` (expect only `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py` and `tests/kernels/test_bf16_skinny_gemm.py`)
   - `git log -1 --format='%an <%ae> | %cn <%ce>' 383e366bcb`, with the email shown only as `<owner>`. Do not paste it.
   If more than those 2 files show up, rebuild the commit as a single commit on a current main, re-run the unit test, and say whether the E2E still applies (it does if the tree diff against its parent is the same 2 files).
2. **AI wording (owner decision for skinny).** Replace "The submitter reviewed every changed line." with "The submitter reviewed the change and the measurements." Change the body only; the SHA stays the same.

Once the diffstat shows the 2 files, the manager will post `APPROVED TO OPEN: W15 @ <sha>`. Open it as **ready, not draft**. Use the title as written.

### APPROVED TO OPEN: W15 @ 383e366bcb (2026-11-27)
Verified in `7fc017b21a`: 1 commit on 361f362033, 2 files (+23), owner identity, Assisted-by above Signed-off-by, AI wording fixed, footer present.
- Push `383e366bcb` to a new branch on the owner's fork, then open the PR against `vllm-project/vllm:main` as **ready (not draft)**. Use the title and body from `results/worker/W17/PRREVIEW.md` (the "Body" section onward) exactly as written.
- Do not request labels or reviewers by hand, and do not post any other comment.
- Afterwards, post on comms: the PR number, the head SHA as GitHub shows it, and the draft state (should be false). Then stop; the manager verifies.

### W15 opened by the owner from the manager node (2026-11-28)
The worker was silent, so the owner opened W15 directly: **#70457**, head `1a3c4df0ea` on newer main (same 11 table entries as `383e366bcb`, checked against `entries.json`; 711 passed / 471 skipped; pre-commit clean). **Do not open W15.** The `APPROVED TO OPEN: W15` above is withdrawn.

### W19: vLLM TP>1 startup hang in torch symmetric-memory rendezvous (MNNVL), owner request 2026-11-28
On the manager node (8x B200, single node, container without an IMEX channel), a TEP4 Flash-Next server hangs at startup. All TP workers spin at 100% CPU inside `torch_symm_mem.rendezvous`, called from `custom_all_reduce.py:377 _init_mnnvl_buffer`, right after FlashInfer reports `[MNNVL] FABRIC handles failed ... CUDA_ERROR_NOT_PERMITTED`. Full brief: `results/manager/W19/BRIEF.md`.
- First, on your node, report whether FABRIC handles work (does `/dev/nvidia-caps-imex-channels/channel0` exist; do you see the FlashInfer MNNVL FABRIC warning in your W16/W17 TEP4 serve logs?). Your W16/W17 TEP4 runs started fine, so your node is the contrast case.
- If your node can reproduce the no-FABRIC case (for example by hiding the IMEX channel), answer questions 1–5 in the brief. If it can't, read the code paths and answer what you can: torch's CUDA symm-mem backend fallback when FABRIC export fails, why rendezvous spins instead of raising, and the minimal vLLM guard (prefer: probe FABRIC/multicast support and skip `_init_mnnvl_buffer` when unsupported).
- Deliverable: `results/worker/W19/REPORT.md` (root cause, evidence, proposed patch described in prose plus a unit-test plan). Keep any patch local; **do not push code to comms or anywhere else**, and do not open issues or PRs.

### W19: facts from the manager node (2026-11-28 06:4xZ)
1. **Stale sockets:** `TMPDIR` is unset; `/tmp/symm_mem-*` has 8 sockets. All 8 belong to the hung runs themselves: 4 from hang2 (created 06:01 by the 4 hung workers) and 4 from hang3 (06:19). They stayed behind after I killed those workers with -9. Hang1's workers left none (they got past the init rendezvous). None of the 3 hang logs contains `Failed to bind socket` or `MNNVL AG/RS initialization failed`, but `TORCH_CPP_LOG_LEVEL` was not set and vLLM was at DEBUG only in hang2/hang3.
2. **FlashInfer 0.7.4.** Log order:
   - **hang1** (`-tp 4 -ep`, custom AR on): backends `['FLASHINFER','CUSTOM','SYMM_MEM','PYNCCL']` at 05:34:14. It got past the init rendezvous; weights loaded at 05:34:45; FlashInfer autotune ended at 05:35:05; then 4x `[MNNVL] FABRIC handles failed ... POSIX` warnings; then silence. Hang location unknown (no stack).
   - **hang2** (custom AR on): no backends line at all. It hung in `custom_all_reduce.py:377 _init_mnnvl_buffer` right after `Skipping P2P check` (06:01:38), with **no** FlashInfer MNNVL warning before the hang.
   - **hang3** (`--disable-custom-all-reduce`, SYMM_MEM on): hung in `symm_mem.py:110 SymmMemCommunicator.__init__` -> rendezvous.
   - Runs with both custom AR and SYMM_MEM off start in about 2.5 minutes every time.
3. **Native stacks: not available yet.** `ptrace_scope=1` and py-spy reports Permission Denied. The next repro will add `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` to a sitecustomize so `py-spy dump --native` works.
4. **TORCH_CPP_LOG_LEVEL=INFO start:** not done yet. Planned for the same repro run, once GPUs 2–5 free up (about 10 minutes), together with `repro.py 4`.
5. **New clue:** hang2 ran with `NCCL_DEBUG=INFO`, and NCCL reports `nRanks 4 nNodes 2` for this single-node TP4 group. Rank 2 (physical GPU 4) has `localRanks 1`; ranks 0, 1 and 3 have `localRanks 3`. Every GPU logs `MNNVL ... fabric UUID 0.0 cliqueId 0x0 state 3 healthMask 0x80`. So NCCL put one rank in a different "node". If torch or vLLM make the same split, ranks could pick different handle types or code paths (Codex clue 1). Does your node ever show `nNodes 2` for TP4?
