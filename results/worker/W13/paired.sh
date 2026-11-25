#!/bin/bash
# W13 paired A/B: usage paired.sh GPU_A GPU_B ROOT "ENV_A" "ENV_B"   (arm names a / b; env strings like "KOPT_AUX_PRIO=-1 KOPT_GATE_FIRST=0")
source ~/kopt-env.sh; cd ~/vllm-kopt; GA=$1; GB=$2; ROOT=$3; EA=$4; EB=$5; mkdir -p $ROOT
M=Qwen/Qwen3.6-35B-A3B-FP8; PA=8871; PB=8872
serve() { env $3 VLLM_DISABLE_COMPILE_CACHE=1 CUDA_VISIBLE_DEVICES=$1 python -m vllm.entrypoints.cli.main serve $M --port $2 --language-model-only \
  --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching > $ROOT/serve_$4.log 2>&1 & }
serve $GA $PA "$EA" a; SA=$!; serve $GB $PB "$EB" b; SB=$!
for P in $PA $PB; do for i in $(seq 1 120); do curl -sf localhost:$P/health >/dev/null && break; sleep 10; done; done
if curl -sf localhost:$PA/health >/dev/null && curl -sf localhost:$PB/health >/dev/null; then
  for K in 1 2 3 4 5 6; do for C in ${CONCS:-1 2 4 8 16}; do
    if [ $((K % 2)) = 1 ]; then ORDER="a b"; else ORDER="b a"; fi
    for ARM in $ORDER; do
      [ $ARM = a ] && P=$PA || P=$PB
      echo "$K $C $ARM $(date -u +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg)" >> $ROOT/log
      python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $P --dataset-name random \
        --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $((C*5)) \
        --max-concurrency $C --num-warmups $C --seed $K --percentile-metrics ttft,tpot,e2el --metric-percentiles 50 \
        --save-result --result-dir $ROOT --result-filename ${ARM}_c${C}_k$K.json > $ROOT/${ARM}_c${C}_k$K.txt 2>&1
    done
  done; done
  [ -n "$AFTER" ] && $AFTER $PA $PB $ROOT
else echo SERVER_FAILED > $ROOT/FAILED; fi
kill -INT $SA $SB; wait $SA $SB 2>/dev/null; sleep 15
echo "DONE $(date -u +%H:%MZ) aot=$(cat $ROOT/serve_*.log | grep -c 'Directly load AOT') envA='$EA' envB='$EB'" >> $ROOT/log
