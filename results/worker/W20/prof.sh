#!/bin/bash
# W20 decode profile (W14 method: torch profiler, skip 24 engine steps, record 32) + one unprofiled TPOT run per c.
# usage: prof.sh NAME TREE "CONCS"   (TEP4 on GPUs 2-5, custom all-reduce on, compile cache off)
source ~/kopt-env.sh; NAME=$1; TREE=$2; CONCS=$3; export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$TREE
M=Qwen/Qwen3.8-Flash-Next-FP8; PORT=8893; OUT=/scratch/kopt-w20/$NAME; mkdir -p $OUT/traces; cd $TREE
python -c "import vllm, vllm.models.qwen4_exp.nvidia.low_latency_gemm as g; print('CHECK vllm from', vllm.__file__); print('CHECK gate plan', sorted(g.QWEN4_EXP_SM100_GEMM_PLANS.get((1,2560),{})))" > $OUT/serve.log 2>&1
echo "CHECK HEAD $(git rev-parse HEAD)" >> $OUT/serve.log
PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
CUDA_VISIBLE_DEVICES=2,3,4,5 setsid python -m vllm.entrypoints.cli.main serve $M --port $PORT -tp 4 --enable-expert-parallel \
  --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 --language-model-only --max-model-len 32768 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching --max-num-seqs 256 --profiler-config "$PCFG" >> $OUT/serve.log 2>&1 &
SPID=$!
for j in $(seq 1 240); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
if curl -sf localhost:$PORT/health >/dev/null; then
  for C in $CONCS; do
    python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $((4*C)) \
      --max-concurrency $C --num-warmups $C --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
      --save-result --result-dir $OUT --result-filename tpot_c$C.json > $OUT/tpot_c$C.txt 2>&1
    python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 128 --random-range-ratio 0 --ignore-eos --num-prompts $C \
      --max-concurrency $C --num-warmups $C --seed 0 --profile \
      --save-result --result-dir $OUT --result-filename prof_c$C.json > $OUT/prof_c$C.txt 2>&1
    sleep 40; mkdir -p $OUT/traces_c$C; for f in $OUT/traces/*; do mv "$f" $OUT/traces_c$C/; done
  done
else echo SERVER_FAILED > $OUT/FAILED; fi
kill -INT $SPID; wait $SPID 2>/dev/null; sleep 10
echo "done $NAME $(date -u +%H:%MZ) aot=$(grep -c 'Directly load AOT' $OUT/serve.log) $(ls $OUT | grep -c FAILED) failed" >> /scratch/kopt-w20/status
