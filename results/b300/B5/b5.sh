#!/bin/bash
# B5: SM103 TP8 AR+RMSNorm fusion cap. Qwen3-32B real weights, TP8, no code edits.
# Phase 1: one DEBUG probe launch per arm (log lines only, no timing).
# Phase 2: timed rounds at INFO, order default 8 16 32, twice. GSM8K in every round-1 arm.
source ~/kopt-env.sh
export VLLM_DISABLE_COMPILE_CACHE=1
M=Qwen/Qwen3-32B; PORT=8852; ROOT=~/b300-work/B5; ARMS=${ARMS:-"default 8 16 32"}; PHASES=${PHASES:-"probe 1 2"}
cd ~/vllm-kopt   # launch from the tree being timed (e66bede56b, clean)
SPID=
cleanup() { local j; kill -INT $SPID 2>/dev/null; for j in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; sleep 5; }
stepback() { echo "$(date +%H:%M:%S) STEPPED_BACK $1" >> $ROOT/status; [ -n "$SPID" ] && cleanup; exit 2; }
others() { ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$1"; }
start() { # $1 arm  $2 outdir  [env...]
  local arm=$1 out=$2; shift 2
  local cc=(); [ "$arm" != default ] && cc=(--compilation-config "{\"pass_config\":{\"fi_allreduce_fusion_max_size_mb\": $arm}}")
  echo "CHECK vllm tree $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) arm=$arm" > $out/serve.log
  env "$@" vllm serve $M --port $PORT -tp 8 --max-num-seqs 512 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching "${cc[@]}" >> $out/serve.log 2>&1 &
  SPID=$!
  local j; for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $SPID 2>/dev/null || break; sleep 10; done
  echo "$(date +%H:%M:%S) SERVER_FAILED $arm $out" >> $ROOT/status; cleanup; exit 1; }
bench() { # $1 out $2 c $3 in $4 outlen $5 N $6 name
  vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
    --random-input-len $3 --random-output-len $4 --random-range-ratio 0 --ignore-eos --num-prompts $5 \
    --max-concurrency $2 --num-warmups 8 --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
    --save-result --result-dir $1 --result-filename $6.json > $1/$6.txt 2>&1; }
for PH in $PHASES; do for ARM in $ARMS; do
  if [ $PH = probe ]; then OUT=$ROOT/probe_$ARM; else OUT=$ROOT/r${PH}_$ARM; fi; mkdir -p $OUT
  others "before $PH $ARM"
  if [ $PH = probe ]; then
    start $ARM $OUT VLLM_LOGGING_LEVEL=DEBUG
  else
    start $ARM $OUT VLLM_LOGGING_LEVEL=INFO
    for C in 32 96 128 192 256 384; do others "$PH $ARM c$C"; bench $OUT $C 512 512 $((2*C)) c$C; done
    others "$PH $ARM ttft"; bench $OUT 4 1024 1 64 ttft_c4
    if [ $PH = 1 ]; then others "$PH $ARM gsm8k"
      python tests/evals/gsm8k/gsm8k_eval.py --port $PORT --use-chat-completions \
        --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64 \
        --save-results $OUT/gsm8k.json > $OUT/gsm8k.txt 2>&1; fi
  fi
  cleanup; SPID=
  echo "$(date +%H:%M:%S) done $PH $ARM" >> $ROOT/status
done; done
echo DONE >> $ROOT/status
