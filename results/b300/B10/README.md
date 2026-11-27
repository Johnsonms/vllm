# B10: #69778 smoke check at e1df7ec5f6

PR head `e1df7ec5f6` (= `c5ef00f633` + `1fe793b18e` + `a6db1de994` + merge of main). Base = merge base `29fa74e654`. Fetched read-only; nothing pushed. 1× B300 per arm, Qwen3.6-35B-A3B-FP8, TP1, `--language-model-only`. Both arms use the same prebuilt extensions; main's csrc changes since that build are removals plus Hy4/FlashKDA only.

**Fusion fires** (`count_fusion.py`, in-process, eager, real weights, 4 prompts × 16 tokens):
- PR: the `(SiluAndMul, kFp8Dynamic128Sym)` producer is called 640 times (40 layers × 16 steps), all on the shared-expert down_proj `(2048, 512)`.
- PR: the CUTLASS block linear gets a pre-quantized input 640 times out of 2560 applies.
- Base: `get_input_quant_key(down_proj)` is None, and there are 0 producer calls.
- Same 16-token greedy text on both arms.

**Paired TPOT** (1 pass × 4 rounds, both servers up, order flipped each round, random 512/512, `--ignore-eos`, median TPOT; AOT count 0 on both):

| c | base ms | PR ms | Δ | pairs |
|---:|---:|---:|---:|---|
| 1 | 2.923 | 2.865 | −1.99% | −2.04..−1.94, 4/4 faster |
| 4 | 3.733 | 3.720 | −0.36% | −0.55..+0.00, 3/4 faster |
| 8 | 4.581 | 4.517 | −1.40% | −2.21..−0.77, 4/4 faster |

**Tests:** `pytest tests/fusion/test_quant_activation_contract.py tests/compile/passes/test_silu_mul_quant_manual_fusion.py -q` gives 56 passed.

**Greedy:** the server greedy check (4 prompts × 64 tokens) matches on 1/4 prompts; the others first diverge at tokens 7, 40 and 50. This is expected: the head carries only v1 (`c5ef00f633`), whose fused quant is not bit-exact with the unfused chain. The bit-exact second commit is not on the branch, which is consistent with B9: v1 is within sampled-GSM8K noise. GSM8K was not rerun here.
