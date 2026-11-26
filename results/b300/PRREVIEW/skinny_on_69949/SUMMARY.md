# Skinny block-FP8 GEMM on #69949 (PRREVIEW)

- **Files:** title in `title.txt`; `body.md` is the exact text to post; diffstat and full SHAs in `diffstat.txt`. The `raw_*` files are the raw outputs quoted in the body.
- **Branch:** owner-local `kopt/skinny-on-69949-v2`, to be pushed as `sm103-skinny-fp8-block-gemm` on the owner's fork.
- **Head:** `fd7eea88349237f939b72052c61b2fb6f1045217` (tree `6dd669471d`). It is one commit on top of #69949's head `d6bccd2252` (2 commits on main).
- **Open as:** draft, once #69949 has a maintainer review and you post `APPROVED TO OPEN`. Mark it ready after #69949 merges; the rebase will need a force-push, which is the owner's call.
- **#69949 now:** REVIEW_REQUIRED. Its only review is a bot (`claude`, COMMENTED).

## PR_RULES / pr-checklist
| Item | Status |
|---|---|
| Identity | `Johnsonms <owner email> \| Johnsonms <owner email>`. Trailers are `Assisted-by: Claude Code` + `Signed-off-by:`, with no Co-authored-by. |
| pre-commit | `uvx pre-commit run --files <5 files>` passes with no changes, including mypy and test tethering. |
| Unit tests | Skinny file: 26 passed in 30.45 s. With `test_block_fp8.py`: PR 394 passed / 1 failed, base 368 passed / 1 failed. The failure is the same on both sides: `test_w8a8_block_fp8_torch_scaled_mm_matmul`, a cuBLASLt `NOT_SUPPORTED` error. |
| E2E | E24 on `fd7eea8834`, TP1 and TP2, c1–c32, 2 GPU-swapped passes × 6 rounds, AOT 0. TP1: −9.03 / −8.75 / −4.44 / −0.15 / +0.25 / +0.16%. TP2: −3.48 / −7.89 / −4.50 / −0.17 / −3.78 / +0.28%. Nothing is outside ±0.4% on the slow side. |
| GSM8K | G12 8+8 (TP1), measured on `07ec1f5c08`: Δ +0.02 pp, t +0.17, flips at the floor, 0/0 systematic. It is carried over per 2b0e393, since the tree differs only in the TP2-only entry and comments. The body says so. |
| Greedy | TP1: single 3/32, batched 200/256 (floors 32/32, 162/256). TP2: 8/32 and 222/256 (floors 32/32, 253/256). Not bit-identical by design: the FP32 summation order differs. |
| Template | Uses `.github/PULL_REQUEST_TEMPLATE.md` sections and its checklist, plus `## AI assistance` and the footer. I dropped the template's last line, because GitHub Actions removes everything below it, which would strip the footer. |

## Points for the manager / owner
1. **TP2 c1 is noisy:** pass 1 −0.27%, pass 2 −6.69%. Both passes are faster, and 10/12 pairs. E22 gave −1.75% here, and E13 (old plan) −6.41%. The body reports the mean (−3.48%) as it came out.
2. **Complexity:** about 500 non-test lines, which is the pr-checklist's "high" band. The body states the size and the justification (8–9% TPOT at TP1 c1/c2).
3. **CI gap:** the new tests skip on the L4 `kernels/quantization` step, and no B200 step runs them. The body documents this. Adding a B200 CI step would be a separate decision; I did not add one.
4. **Not run:** `VLLM_GPU_SYNC_CHECK=error`. The added per-call work is a cached dict lookup and does no host read of GPU data. I can run it if you want.
5. Raw data on the node: `e2e/E24_tp{1,2}`, `e2e/E23_*`, `greedy/E24_greedy_tp{1,2}`, `gsm/G12_sk_h{1,2}`, `ut_E24_*.txt`.
