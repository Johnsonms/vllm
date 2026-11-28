#!/bin/bash
# usage: prof.sh NAME MODEL GPU PORT <serve args...>  -> kernel selection + profiles at c1/c32 decode and 4096-token prefill
source ~/kopt-env.sh; source ~/kopt-work/killtree.sh; NAME=$1; M=$2; GPU=$3; PORT=$4; shift 4
OUT=~/kopt-work/H4/prof/$NAME; rm -rf $OUT; mkdir -p $OUT/traces; cd ~/vllm-kopt
[ -f ~/kopt-work/STOP ] && exit 1
PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
VLLM_LOGGING_LEVEL=DEBUG CUDA_VISIBLE_DEVICES=$GPU taskset -c 64-127 vllm serve $M --port $PORT --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
  --max-num-seqs 256 --max-model-len 8192 --profiler-config "$PCFG" "$@" > $OUT/serve.log 2>&1 &
SP=$!
for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 10; done
if curl -sf localhost:$PORT/health >/dev/null; then
  run() { # tag in out c n profile
    vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len $2 --random-output-len $3 --random-range-ratio 0 --ignore-eos --num-prompts $5 \
      --max-concurrency $4 --num-warmups $4 --seed 0 $6 --save-result --result-dir $OUT --result-filename $1.json > $OUT/$1.txt 2>&1
  }
  for C in 1 32; do
    run tpot_c$C 512 512 $C $((4*C>16?4*C:16)) ""
    run prof_c$C 512 128 $C $C --profile
    sleep 40; mkdir -p $OUT/traces_c$C; mv $OUT/traces/* $OUT/traces_c$C/ 2>/dev/null
  done
  # prefill: 4096-token prompts, 1 output token; profile spans 32 engine steps after 24 skipped
  run prof_prefill 4096 1 1 64 --profile
  sleep 40; mkdir -p $OUT/traces_prefill; mv $OUT/traces/* $OUT/traces_prefill/ 2>/dev/null
else echo SERVER_FAILED > $OUT/FAILED; fi
stop_server $SP
echo "done $NAME $(date -u +%H:%MZ)" >> ~/kopt-work/H4/prof/status
