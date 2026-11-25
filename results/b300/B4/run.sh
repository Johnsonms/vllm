#!/bin/bash
# B4: 2 sets x 4 seeds = 8 processes, one per GPU. Arms: cuBLAS vs given SM100 config (column "sm100").
source ~/kopt-env.sh; cd ~/vllm-kopt
W=~/b300-work/B4
~/b300-work/gpu_others.sh || { echo STEPPED_BACK; exit 2; }
g=0
for set in dense_qwen3 qwen3_moe; do for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python $W/../B2/verify_sm103.py $W/empty.json $W/${set}_sm100.json $W/verify_${set}_s$s.jsonl $s > $W/log_${set}_s$s.txt 2>&1 &
  g=$((g+1)); done; done
while pgrep -u $USER -f verify_sm103 >/dev/null; do
  ~/b300-work/gpu_others.sh >/dev/null || { echo "$(date +%H:%M) STEPPED_BACK"; pkill -u $USER -f verify_sm103; exit 2; }
  sleep 30; done
echo DONE
