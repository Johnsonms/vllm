#!/bin/bash
# W13 profile: usage prof.sh ARM GPU PORT PRIO GATEFIRST ; B6 method (torch profiler, skip 24 engine steps, record 32), c = 1 4 8
source ~/kopt-env.sh; export VLLM_DISABLE_COMPILE_CACHE=1
ARM=$1; GPU=$2; PORT=$3; export KOPT_AUX_PRIO=$4 KOPT_GATE_FIRST=$5
M=Qwen/Qwen3.6-35B-A3B-FP8; OUT=/scratch/kopt-w13/prof/$ARM; mkdir -p $OUT/traces; cd ~/vllm-kopt
PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.cli.main serve $M --port $PORT --language-model-only --max-num-seqs 256 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --no-enable-prefix-caching --profiler-config "$PCFG" > $OUT/serve.log 2>&1 &
SPID=$!
for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
if curl -sf localhost:$PORT/health >/dev/null; then
  for C in 1 4 8; do
    python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 128 --random-range-ratio 0 --ignore-eos --num-prompts $C \
      --max-concurrency $C --num-warmups $C --seed 0 --profile \
      --save-result --result-dir $OUT --result-filename prof_c$C.json > $OUT/prof_c$C.txt 2>&1
    sleep 30; mkdir -p $OUT/traces_c$C; for f in $OUT/traces/*; do mv "$f" $OUT/traces_c$C/; done
  done
else echo SERVER_FAILED > $OUT/FAILED; fi
kill -INT $SPID; wait $SPID 2>/dev/null
echo "done $ARM $(date -u +%H:%MZ)" >> /scratch/kopt-w13/prof/status
