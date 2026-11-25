#!/bin/bash
# W14 decode profile (B1/B6 method: torch profiler, skip 24 engine steps, record 32), plus one unprofiled TPOT run per c.
# usage: prof.sh NAME MODEL GPUS PORT "CONCS" <serve args...>
source ~/kopt-env.sh; export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$HOME/vllm-kopt-w14
NAME=$1; M=$2; GPUS=$3; PORT=$4; CONCS=$5; shift 5
OUT=/scratch/kopt-w14/$NAME; mkdir -p $OUT/traces; cd ~/vllm-kopt-w14
python -c "import vllm, vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans as p; print('CHECK vllm from', vllm.__file__); print('CHECK (10, 0) plans:', (10, 0) in p.SKINNY_GEMM_PLANS_BY_CAPABILITY)" > $OUT/serve.log 2>&1
git rev-parse HEAD >> $OUT/serve.log
PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.cli.main serve $M --port $PORT --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --max-num-seqs 256 --profiler-config "$PCFG" "$@" >> $OUT/serve.log 2>&1 &
SPID=$!
for j in $(seq 1 240); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
if curl -sf localhost:$PORT/health >/dev/null; then
  for C in $CONCS; do
    N=$(( 4*C > 16 ? 4*C : 16 ))
    python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $N \
      --max-concurrency $C --num-warmups $C --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
      --save-result --result-dir $OUT --result-filename tpot_c$C.json > $OUT/tpot_c$C.txt 2>&1
    python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 128 --random-range-ratio 0 --ignore-eos --num-prompts $C \
      --max-concurrency $C --num-warmups $C --seed 0 --profile \
      --save-result --result-dir $OUT --result-filename prof_c$C.json > $OUT/prof_c$C.txt 2>&1
    sleep 40; mkdir -p $OUT/traces_c$C; for f in $OUT/traces/*; do mv "$f" $OUT/traces_c$C/; done
  done
else echo SERVER_FAILED > $OUT/FAILED; fi
kill -INT $SPID; wait $SPID 2>/dev/null
echo "done $NAME $(date -u +%H:%MZ)" >> /scratch/kopt-w14/status
