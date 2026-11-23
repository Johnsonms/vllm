# Reproduce the manager's environment

The manager's node runs 8x NVIDIA B200 (driver 595.91.07) with CUDA 13.2, torch 2.14.0+cu132, vLLM `0.34.0rc1.dev1161+g8399da378`, FlashInfer 0.7.4 and Triton 3.8.0. Use the same vLLM base commit so the numbers are comparable.

```bash
# 1. vLLM at the round's base commit
git clone https://github.com/vllm-project/vllm.git ~/vllm-kopt
cd ~/vllm-kopt && git checkout 8399da37800dfa13d19e116cb2c838aa33538320

# 2. venv (Python 3.12) + precompiled install
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in cupti-python ninja
vllm download-kernels           # FlashInfer cubins; without them FlashInfer kernels are disabled

# 3. env file, source it before every run (~/kopt-logs/env.sh on the manager)
cat > ~/kopt-env.sh <<'EOF'
source $HOME/vllm-kopt/.venv/bin/activate
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH CUDA_HOME=/usr/local/cuda
export HF_HOME=<your HF cache dir>        # keep the HF token out of this repo
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
EOF
```

## Pitfalls the manager hit

- **torchrun rendezvous timeout:** pass `--master-addr=127.0.0.1` explicitly.
- **NCCL "No route to host":** set `NCCL_SOCKET_IFNAME=lo` and `GLOO_SOCKET_IFNAME=lo` on a single node.
- **OOM in `vllm bench forward`:** add `--gpu-memory-utilization 0.7 --max-model-len 16384`.
- **`pkill -f <pattern>` / `pgrep -f <pattern>` matching your own shell:** use the `[x]yz` bracket trick.
- **CUPTI timing:** use `flashinfer.testing.bench_gpu_time_with_cupti(..., use_cuda_graph=True, cold_l2_cache=True, input_args=...)`. Pass tensors through `input_args`, or cold-L2 is silently off.
