#!/bin/bash
# W16: one server start of one arm, TEP4 on GPUs 2-5. usage: start.sh TREE OUTDIR "CONCS" [extra script run before shutdown]
source ~/kopt-env.sh; TREE=$1; OUT=$2; CONCS=$3; AFTER=$4; mkdir -p $OUT; cd $TREE
export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$TREE
M=Qwen/Qwen3.8-Flash-Next-FP8; PORT=8881
python -c "import vllm, vllm.models.qwen4_exp.nvidia.low_latency_gemm as g; print('CHECK vllm from', vllm.__file__); print('CHECK table file', g.__file__); print('CHECK SM100 entries', sum(len(v) for v in g.QWEN4_EXP_SM100_GEMM_PLANS.values()))" > $OUT/serve.log 2>&1
echo "CHECK groups=${KOPT_W16_GROUPS-unset}" >> $OUT/serve.log; echo "CHECK HEAD $(git rev-parse HEAD) dirty=$(git status --short | grep -vc '^??')" >> $OUT/serve.log
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m vllm.entrypoints.cli.main serve $M --port $PORT -tp 4 --enable-expert-parallel \
  --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --language-model-only --max-model-len 32768 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching --max-num-seqs 256 >> $OUT/serve.log 2>&1 &
SPID=$!
for j in $(seq 1 240); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
if curl -sf localhost:$PORT/health >/dev/null; then
  # warm-up: one 16-way burst, discarded
  python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
    --random-input-len 512 --random-output-len 256 --random-range-ratio 0 --ignore-eos --num-prompts 32 --max-concurrency 16 --seed 99 > $OUT/warm.txt 2>&1
  for K in 1 2 3 4; do for C in $CONCS; do
    echo "$K $C $(date -u +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg)" >> $OUT/log
    python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $((C*5)) \
      --max-concurrency $C --num-warmups $C --seed $K --percentile-metrics ttft,tpot,e2el --metric-percentiles 50 \
      --save-result --result-dir $OUT --result-filename c${C}_k$K.json > $OUT/c${C}_k$K.txt 2>&1
  done; done
  [ -n "$AFTER" ] && $AFTER $PORT $OUT
else echo SERVER_FAILED > $OUT/FAILED; fi
kill -INT $SPID; wait $SPID 2>/dev/null; sleep 15
echo "DONE $(date -u +%H:%MZ) aot=$(grep -c 'Directly load AOT' $OUT/serve.log)" >> $OUT/log
