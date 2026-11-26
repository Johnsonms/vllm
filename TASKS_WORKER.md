
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
