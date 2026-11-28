#!/bin/bash
# W23: start one TP1 server. usage: serve.sh ARM(base|mx) GPU PORT OUTDIR   (runs in foreground; caller backgrounds it)
source ~/kopt-env.sh; ARM=$1; GPU=$2; PORT=$3; OUT=$4; TREE=$HOME/vllm-kopt-w16a; mkdir -p $OUT; cd $TREE
export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$TREE
M=Qwen/Qwen3-8B-FP8
if [ $ARM = mx ]; then export KOPT_MX_MLP=1 KOPT_MX_SNAPSHOT=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/*) KOPT_MX_MIN_M=${KOPT_MX_MIN_M:-16} KOPT_MX_LOG=$OUT/mx_hits.log; fi
echo "CHECK arm=$ARM gpu=$GPU HEAD $(git rev-parse HEAD) KOPT_MX_MLP=${KOPT_MX_MLP-unset}" > $OUT/serve.log
CUDA_VISIBLE_DEVICES=$GPU exec python -m vllm.entrypoints.cli.main serve $M --port $PORT --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching --max-num-seqs 256 >> $OUT/serve.log 2>&1
