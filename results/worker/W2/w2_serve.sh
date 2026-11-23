#!/bin/bash
# W2 (revised): fusion-cap A/B with real serving (torch.compile on). Qwen3-32B TP8, dummy weights.
# Adapted from results/manager/m2b_serve.sh. Arms: default (1 MB) / 16 / 64 MB; 2 rounds alternating.
# DEBUG logging is on in every arm so that each serve.log shows "Flashinfer Allreduce Fusion: <max_token_num>".
# Each run first checks that no GPU shows utilisation; if one does, the arm is skipped and marked NOT_IDLE.
source ~/kopt-env.sh; cd ~/vllm-kopt; ROOT=~/kopt-comms/results/worker/W2; mkdir -p $ROOT; PORT=8821
M=Qwen/Qwen3-32B
export VLLM_LOGGING_LEVEL=DEBUG
for R in 1 2; do for CAP in default 16 64; do
  OUT=$ROOT/cap${CAP}_r$R; mkdir -p $OUT
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > $OUT/nvidia_smi_before.txt
  if awk -F', ' '{if ($2+0 > 0) b=1} END{exit !b}' $OUT/nvidia_smi_before.txt; then echo NOT_IDLE > $OUT/FAILED; echo "cap$CAP r$R NOT_IDLE" >> $ROOT/status; continue; fi
  if [ $CAP = default ]; then CC=(); else CC=(--compilation-config "{\"pass_config\":{\"fi_allreduce_fusion_max_size_mb\": $CAP}}"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $M --port $PORT -tp 8 --load-format dummy \
    --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching "${CC[@]}" > $OUT/serve.log 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 15; done
  if curl -sf localhost:$PORT/health >/dev/null; then
    for C in 1 4; do   # prefill-heavy: 8k in, 16 out
      vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
        --random-input-len 8192 --random-output-len 16 --random-range-ratio 0 --ignore-eos --num-prompts $((C*8)) \
        --max-concurrency $C --num-warmups 2 --seed 1 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50 \
        --save-result --result-dir $OUT --result-filename pre_c$C.json > $OUT/pre_c$C.txt 2>&1
    done
    for C in 16 64 256; do   # decode-heavy: 512 in, 512 out
      vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
        --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $((C*3)) \
        --max-concurrency $C --num-warmups $((C<16?C:16)) --seed 0 --percentile-metrics ttft,tpot,e2el --metric-percentiles 50 \
        --save-result --result-dir $OUT --result-filename dec_c$C.json > $OUT/dec_c$C.txt 2>&1
    done
  else echo SERVER_FAILED > $OUT/FAILED; fi
  kill $SPID; wait $SPID 2>/dev/null; sleep 20
  echo "cap$CAP r$R done" >> $ROOT/status
done; done
echo DONE >> $ROOT/status
