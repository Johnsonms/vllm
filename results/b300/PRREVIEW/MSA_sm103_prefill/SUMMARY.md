# MSA — SM103 Blackwell sparse prefill: hardware exp2 + ld.red (PRREVIEW)

- **Target:** `vllm-project/MSA`, base `dev` `487773e` (unchanged since branching).
- **Head:** `311fa9d2d1`. Only the message was amended from 4bc7377, with the graph-mode numbers; the code is identical.
- **Files:** `title.txt`, `body.md`, `diffstat.txt`.

## Checklist
| Item | Status |
|---|---|
| Dup check (MSA PRs + issues: ex2, ld.red/ldred, SM103/sm_103, B300, exp2 emulation, blackwell_prefill) | No PR or issue about ex2 or ld.red. Open PRs #46/#47/#49 do not touch `blackwell_prefill`. #48 (merged) uses reducing TMEM loads in the Q8KV8 indexer, not this kernel. |
| Repo conventions | No CONTRIBUTING and no `.github` templates. Merged PRs use `[Perf][Attention]` titles, a `## Changes / ## Correctness / ## Performance` body, and most commits carry Signed-off-by. This PR follows that. |
| Identity | Johnsonms (owner) author and committer, Signed-off-by, no Co-authored-by |
| Tests | `test_sparse_atten.py` on SM103: 1173 passed / 143 skipped, same as `dev` |
| Numerics | ~5% of bf16 outputs differ by 1 ulp (hardware exp2); fp32-reference error unchanged (< 1e-8 difference) |
| Microbench | CUDA graph + cold L2, 3 alternating rounds: −6.9..−12.9% |
| FA4 precedent | 8e0b5d7 (ex2 off on SM103), #2696 (ld.red row max) |
| Not measured | E2E (no MiniMax-M3 weights), SM100, SM107 |

## Needs a decision before opening
1. **The owner has no MSA fork** (`Johnsonms/MSA` does not exist). Opening needs one. Creating a fork under the owner's account is a new outward action: please confirm it is covered by the delegation.
2. **Attribution footer:** same question as B1 (PR_RULES §5.5 footer vs the owner's earlier instruction). AI-assistance section added per 5f97b92.
