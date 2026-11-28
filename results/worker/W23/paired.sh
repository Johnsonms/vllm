#!/bin/bash
# W23 paired E2E: base and mx servers up together (TP1, one GPU each), bursts alternate. usage: paired.sh GPU_BASE GPU_MX ROOT PORT0
source ~/kopt-env.sh; GB=$1; GM=$2; ROOT=$3; PB=$4; PM=$(($4+1)); mkdir -p $ROOT; M=Qwen/Qwen3-8B-FP8; cd $HOME/vllm-kopt-w16a; export PYTHONPATH=$HOME/vllm-kopt-w16a
setsid /scratch/kopt-w23/serve.sh base $GB $PB $ROOT/srv_base & SB=$!
setsid /scratch/kopt-w23/serve.sh mx $GM $PM $ROOT/srv_mx & SM=$!
for P in $PB $PM; do for i in $(seq 1 120); do curl -sf localhost:$P/health >/dev/null && break; sleep 10; done; done
bench() { python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --dataset-name random --random-range-ratio 0 --ignore-eos "$@"; }
if curl -sf localhost:$PB/health >/dev/null && curl -sf localhost:$PM/health >/dev/null; then
  for ARM in base mx; do [ $ARM = base ] && P=$PB || P=$PM   # discarded warm-up on both
    bench --port $P --random-input-len 512 --random-output-len 128 --num-prompts 64 --max-concurrency 32 --seed 99 > $ROOT/warm_$ARM.txt 2>&1
    bench --port $P --random-input-len 4096 --random-output-len 1 --num-prompts 8 --max-concurrency 4 --seed 99 >> $ROOT/warm_$ARM.txt 2>&1; done
  for K in 1 2 3 4 5 6; do for W in d1 d8 d32 d128 p1 p4 p16; do
    C=${W:1}; [ $((K % 2)) = 1 ] && ORDER="base mx" || ORDER="mx base"
    for ARM in $ORDER; do [ $ARM = base ] && P=$PB || P=$PM
      echo "$K $W $ARM $(date -u +%H:%M:%S) load=$(cut -d' ' -f1 /proc/loadavg)" >> $ROOT/log
      if [ ${W:0:1} = d ]; then
        bench --port $P --random-input-len 512 --random-output-len 256 --num-prompts $((C*5)) --max-concurrency $C --num-warmups $C --seed $K \
          --percentile-metrics ttft,tpot,e2el --metric-percentiles 50 --save-result --result-dir $ROOT --result-filename ${ARM}_${W}_k$K.json > $ROOT/${ARM}_${W}_k$K.txt 2>&1
      else
        bench --port $P --random-input-len 4096 --random-output-len 1 --num-prompts $((C*8)) --max-concurrency $C --num-warmups $C --seed $K \
          --percentile-metrics ttft,e2el --metric-percentiles 50 --save-result --result-dir $ROOT --result-filename ${ARM}_${W}_k$K.json > $ROOT/${ARM}_${W}_k$K.txt 2>&1
      fi
    done
  done; done
  for RUN in base_r1 mx_r1 mx_r2 base_r2; do ARM=${RUN%_*}; [ $ARM = base ] && P=$PB || P=$PM; mkdir -p $ROOT/gsm_$RUN
    python tests/evals/gsm8k/gsm8k_eval.py --port $P --use-chat-completions --max-tokens 1024 --temperature 0 \
      --chat-template-kwargs '{"enable_thinking": false}' --max-concurrency 64 --save-results $ROOT/gsm_$RUN/result.json > $ROOT/gsm_$RUN/eval.txt 2>&1
    echo "$RUN $(date -u +%H:%MZ) $(grep -a -i 'accuracy' $ROOT/gsm_$RUN/eval.txt | tail -n 1)" >> $ROOT/log
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $GB,$GM > $ROOT/mem.txt
else echo SERVER_FAILED > $ROOT/FAILED; fi
kill -INT -- -$SB -$SM 2>/dev/null; sleep 20
echo "DONE $ROOT $(date -u +%H:%MZ) failed=$(ls $ROOT | grep -c FAILED)" >> /scratch/kopt-w23/status
