#!/bin/bash
# W21 step 1: TP1 serve profile of one block-128 FP8 model. usage: prof.sh MODEL GPU PORT
# torch profiler (skip 24 engine steps, record 32) at c1, c32 and prefill 4096; one unprofiled TPOT run per c.
source ~/kopt-env.sh; M=$1; GPU=$2; PORT=$3; TREE=$HOME/vllm-kopt-w16a
export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$TREE
OUT=/scratch/kopt-w21/prof/$(basename $M); mkdir -p $OUT/traces; cd $TREE
echo "CHECK HEAD $(git rev-parse HEAD) dirty=$(git status --short | grep -vc '^??')" > $OUT/serve.log
prof() { echo "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":$1,\"max_iterations\":$2,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"; }
bench() { python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random --random-range-ratio 0 --ignore-eos --seed 0 "$@"; }
CUDA_VISIBLE_DEVICES=$GPU setsid python -m vllm.entrypoints.cli.main serve $M --port $PORT --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching --max-num-seqs 256 --profiler-config "$(prof 24 32)" >> $OUT/serve.log 2>&1 &
SPID=$!
for j in $(seq 1 240); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
if curl -sf localhost:$PORT/health >/dev/null; then
  for C in 1 32; do
    bench --random-input-len 512 --random-output-len 512 --num-prompts $((4*C)) --max-concurrency $C --num-warmups $C \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 --save-result --result-dir $OUT --result-filename tpot_c$C.json > $OUT/tpot_c$C.txt 2>&1
    bench --random-input-len 512 --random-output-len 128 --num-prompts $C --max-concurrency $C --num-warmups $C --profile > $OUT/prof_c$C.txt 2>&1
    sleep 30; mkdir -p $OUT/traces_c$C; for f in $OUT/traces/*; do mv "$f" $OUT/traces_c$C/; done
  done
  # prefill 4096: 40 sequential single-prompt requests, 1 output token; profiler skips 24 steps so ~12 prefill steps are recorded
  bench --random-input-len 4096 --random-output-len 1 --num-prompts 8 --max-concurrency 1 --num-warmups 2 \
    --percentile-metrics ttft,e2el --metric-percentiles 50 --save-result --result-dir $OUT --result-filename ttft_p4096.json > $OUT/ttft_p4096.txt 2>&1
  bench --random-input-len 4096 --random-output-len 1 --num-prompts 40 --max-concurrency 1 --profile > $OUT/prof_p4096.txt 2>&1
  sleep 30; mkdir -p $OUT/traces_p4096; for f in $OUT/traces/*; do mv "$f" $OUT/traces_p4096/; done
else echo SERVER_FAILED > $OUT/FAILED; fi
kill -INT $SPID; wait $SPID 2>/dev/null
echo "done $M $(date -u +%H:%MZ) $(ls $OUT | grep -c FAILED) failed" >> /scratch/kopt-w21/status
