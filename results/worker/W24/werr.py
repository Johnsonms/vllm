"""Per-layer weight error vs the checkpoint's own values (FP8 x fp32 128x128 scale):
 dg = what vLLM/DeepGEMM runs on SM100 (requant_weight_ue8m0_inplace: power-of-two 128x128 scales)
 mx = prototype (MXFP8, power-of-two 1x32 scales along K)."""
import json, sys, torch, inspect
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils import fp8_utils
from quack.blockscaled import BlockScaledOperand, MXFP8_E4M3
snap = sys.argv[1]; dev = "cuda"
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(f"{snap}/{idx[name]}", "pt", device=dev) as f: return f.get_tensor(name)
ex = lambda ws, n, k: ws.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(inspect.signature(fp8_utils.requant_weight_ue8m0_inplace), flush=True)
tot = {}
for l in range(36):
    row = dict(layer=l)
    for name in ("gate_proj", "up_proj", "down_proj"):
        p = f"model.layers.{l}.mlp.{name}."
        wq = load(p + "weight"); ws = load(p + "weight_scale_inv").float(); n, k = wq.shape
        w0 = wq.float() * ex(ws, n, k)
        wq2, ws2 = wq.clone(), ws.clone(); fp8_utils.requant_weight_ue8m0_inplace(wq2, ws2, block_size=(128, 128))
        wdg = wq2.float() * ex(ws2.float(), n, k)
        wmx = BlockScaledOperand.quantize(w0, MXFP8_E4M3).dequantize().float()
        row[name] = (round(rel(wdg, w0), 5), round(rel(wmx, w0), 5), bool(((ws2.log2() % 1) == 0).all()))
    print(json.dumps(row), flush=True)
