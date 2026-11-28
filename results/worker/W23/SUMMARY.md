# W23: end-to-end MXFP8 MLP prototype on Qwen3-8B-FP8 (B200, TP1)

> **Correction 2026-11-28 18:43Z (see `results/worker/W24/SUMMARY.md`):** the vLLM arm of the kernel benches in this file ran a slower activation-quant path than serving (two Inductor kernels + a DeepGEMM scale-pack kernel instead of the one packed kernel): +5.3 us per GEMM on the vLLM side. Kernel speedups below M = 128 are overstated (roughly no gain there); M >= 512 is affected by a few percent. End-to-end numbers are not affected.

2026-11-28 18:19Z. vLLM main 361f362033 + local env-gated prototype (code not pushed), quack-kernels 0.6.5, flashinfer 0.7.4. Base arm = same tree with the env switch off.

## Verdict
- **It works end to end and the gain survives, but it is smaller than the kernel numbers, and GSM8K drops by 0.4 points.**
- **Prefill (4096 tokens): TTFT -4.2 % (c1), -7.8 % (c4), -8.0 % (c16)**, 35 of 36 paired rounds faster.
- **Decode: TPOT -1.4 % (c32), -2.8 % (c128)**, 22 of 24 rounds faster. c1 and c8 stay on the DeepGEMM path (threshold 16): -0.6 % and -0.3 %, which I treat as no change.
- **GSM8K: 0.9067 (base, 4 of 4 runs) vs 0.9030 / 0.9030 / 0.9030 / 0.9022 (prototype).** That is 5-6 questions of 1,319. One run's sampling error is about 0.8 points, so the drop is not significant, but it has the same sign in every run and I cannot call the change lossless.
- Memory: +5.6 GB (155.7 MB per layer x 36), because the prototype keeps both weight copies.

## What the prototype does
For M >= 16 rows, each MLP runs 3 kernels instead of 5:
1. FlashInfer `mxfp8_quantize` on the BF16 input (1x32 UE8M0 scales in the 128x4 swizzled layout, which is QuACK's blocked layout byte for byte);
2. QuACK `swiglu_quant_mod`: gate_up GEMM + SwiGLU + MXFP8 output with scales, one kernel;
3. QuACK MXFP8 GEMM for down_proj, BF16 out.
Below 16 rows it calls vLLM's DeepGEMM path unchanged. At first use each layer re-reads its block-128 weights from the checkpoint, dequantizes them and re-quantizes to MXFP8 (gate / up rows interleaved). Tile configs are pinned per M range for both kernels (never tile_n=192). The whole MLP is one opaque custom op, so it is captured in CUDA graphs for decode and runs eagerly for prefill. Path hits were logged: every CUDA-graph size >= 16 and every prefill size took the MXFP8 path.

## E2E (clean run "v3": 2 passes x 6 rounds, both servers up, bursts alternate, base on GPU 2 / prototype on GPU 3 in pass 1, base on 5 / prototype on 4 in pass 2)
| workload | metric | pass 1 base | pass 1 proto | delta | pass 2 base | pass 2 proto | delta | mean delta | rounds proto faster |
|---|---|---|---|---|---|---|---|---|---|
| decode c1 | TPOT ms | 3.458 | 3.418 | -1.16 % | 3.469 | 3.468 | -0.03 % | -0.59 % | 10/12 |
| decode c8 | TPOT ms | 3.594 | 3.572 | -0.60 % | 3.602 | 3.602 | +0.01 % | -0.30 % | 9/12 |
| decode c32 | TPOT ms | 4.191 | 4.119 | -1.71 % | 4.146 | 4.104 | -1.02 % | -1.36 % | 11/12 |
| decode c128 | TPOT ms | 6.628 | 6.385 | -3.67 % | 6.558 | 6.427 | -1.99 % | -2.83 % | 11/12 |
| prefill 4096/1 c1 | TTFT ms | 52.512 | 49.783 | -5.20 % | 50.603 | 48.955 | -3.26 % | -4.23 % | 11/12 |
| prefill 4096/1 c4 | TTFT ms | 144.646 | 133.396 | -7.78 % | 142.290 | 131.315 | -7.71 % | -7.75 % | 12/12 |
| prefill 4096/1 c16 | TTFT ms | 563.513 | 518.638 | -7.96 % | 546.990 | 503.344 | -7.98 % | -7.97 % | 12/12 |

Decode: 512 in / 256 out, 5 x c prompts per burst. Prefill: 4096 in / 1 out, 8 x c prompts. The two passes do not use the same GPU pair swapped (2,3 and 5,4); the prototype's GPU index is the higher one in pass 1 and the lower one in pass 2. Raw files in `e2e/`.

**Decode gain is well below the kernel estimate.** The warm kernel sweep gives 1.23x for the MLP at M = 32, and the MLP is about 39 % of kernel time, which would be about 7 % of TPOT; measured -1.4 %. I have not explained the gap. Candidates, not tested: step time at c32 / c128 that is not GPU kernel time (4.19 ms TPOT vs 3.81 ms kernel time at c32), and kernel behaviour inside the full graph differing from the 2-layer bench. A c32 profile of the v2 prototype is the next check.

## How we got here (two things went wrong first)
1. **v1 prototype (one pinned config, threshold 8): decode was slower**, +22.9 % (c8), +15.8 % (c32), +7.2 % (c128); prefill was already -5.8 / -8.3 / -9.0 %. Cause: tile 128x256 for down_proj (N = 4096 gives 16 tiles), and a threshold taken from cold-L2 numbers.
2. **The cold-L2 microbench is biased at small M.** Without the L2 flush the vLLM chain at M = 8 is 46.9 us per layer (61.1 us cold) and the prototype 52.8 us (54.4 us). The serve profile agrees with the warm numbers. The W21 and step-0 speedups at M <= 8 are therefore too optimistic.
3. **A first v2 E2E run is discarded**: by my mistake two copies of the test ran at once on the same GPUs and ports. The table above is a clean rerun.

## Warm kernel sweep that set the configs (`multilayer.py`, `multi.jsonl`)
One CUDA graph runs the MLP of 2 consecutive layers with distinct real weights, nothing flushed; us per layer; 12 configs swept for each kernel.

| M | vLLM | prototype (best config) | speedup |
|---|---|---|---|
| 8 | 46.9 | 52.8 | 0.89 |
| 16 | 50.0 | 44.1 | 1.13 |
| 32 | 52.3 | 42.4 | 1.23 |
| 64 | 53.7 | 42.4 | 1.27 |
| 128 | 58.4 | 42.3 | 1.38 |
| 256 | 67.4 | 47.0 | 1.44 |
| 512 | 86.6 | 72.8 | 1.19 |
| 1024 | 148.7 | 123.7 | 1.20 |
| 2048 | 279.8 | 218.6 | 1.28 |
| 4096 | 512.0 | 436.6 | 1.17 |

Configs used (gated / down): M < 256: 128x256 / 128x64; 256-511: 256x128 2-CTA / 128x64; 512-2047: 256x128 2-CTA / 256x128 2-CTA; 2048-3071: 256x128 2-CTA / 256x256 2-CTA; >= 3072: 256x256 2-CTA / 256x256 2-CTA. The same bench with 8 layers in one graph crashed with a CUDA launch failure inside DeepGEMM; I did not investigate and used 2 layers.

## Other measurements
- FlashInfer `mxfp8_quantize` output equals QuACK's own quantizer bit for bit (values and scales) at all 8 sizes tested.
- Host cost of the prototype path outside CUDA graphs: 75 us per MLP call, plus 0.2-0.3 ms the first time a new M is seen. Sizes that are not multiples of 128 work.
- Accuracy at kernel level (random activations): MLP output error vs FP32 0.059 for both arms; after 2 chained layers the two arms differ by 0.05 relative.

## What a real landing would need
**QuACK**
1. Fix quantized output with tile_n=192 (W21 bug), or exclude it from the tuner.
2. A config choice that does not need runtime autotuning per M (prefill sees arbitrary M). The prototype uses a hand table.
3. Support for 1x128 activation / 128x128 weight FP8 scales would remove the re-quantization, the second weight copy and probably the GSM8K question. Today only MXFP8 1x32 exists.
4. `concat_layout` for blockscaled gated GEMMs, to avoid re-laying out the merged gate_up weight.

**vLLM**
1. A linear-kernel class for block-FP8 that owns MXFP8 weights (instead of the prototype's side copy and checkpoint re-read), with the DeepGEMM path kept for small M, or a second copy accepted (+5.6 GB on 8B).
2. An MLP-level fused op: today gate_up, activation and down_proj are separate layers and the quantize step belongs to each linear. The gain needs the three steps in one call.
3. Torch-compile integration (the prototype is an opaque op with a fake impl) and TP > 1 (not tested; down_proj is row-parallel, the all-reduce follows it).
4. Accuracy sign-off on more than GSM8K, because the weights change.

**Not done:** Qwen3-32B, TP > 1, c > 128, output-quality tests other than GSM8K, a profile of the v2 prototype.
