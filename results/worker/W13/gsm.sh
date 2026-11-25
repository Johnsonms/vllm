#!/bin/bash
# usage: gsm.sh NAME GPU PORT PRIO
source ~/kopt-env.sh; cd ~/vllm-kopt; M=Qwen/Qwen3.6-35B-A3B-FP8; OUT=/scratch/kopt-w13/gsm8k/$1; mkdir -p $OUT
KOPT_AUX_PRIO=$4 VLLM_DISABLE_COMPILE_CACHE=1 CUDA_VISIBLE_DEVICES=$2 python -m vllm.entrypoints.cli.main serve $M --port $3 --language-model-only \
  --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching > $OUT/serve.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -sf localhost:$3/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
python tests/evals/gsm8k/gsm8k_eval.py --port $3 --use-chat-completions --max-tokens 1024 --temperature 0 \
  --chat-template-kwargs '{"enable_thinking": false}' --max-concurrency 64 --save-results $OUT/result.json > $OUT/eval.txt 2>&1
kill -INT $SPID; wait $SPID 2>/dev/null
echo "$1 $(date -u +%H:%MZ) $(grep -a -i 'accuracy' $OUT/eval.txt | tail -n 1)" >> /scratch/kopt-w13/gsm8k/status
