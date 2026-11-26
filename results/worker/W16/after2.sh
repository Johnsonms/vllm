#!/bin/bash
# greedy check, then GSM_N GSM8K runs with per-question dumps. args: PORT OUTDIR
python3 /scratch/kopt-w16/greedy.py $1 $2 > $2/greedy.txt 2>&1
for i in $(seq 1 ${GSM_N:-0}); do
  GSM_DUMP=$2/gsm_perq_$i.json python /scratch/kopt-w13/gsm_dump.py --port $1 --use-chat-completions --max-tokens 1024 --temperature 0 \
    --chat-template-kwargs '{"enable_thinking": false}' --max-concurrency 64 --save-results $2/gsm_result_$i.json > $2/gsm_eval_$i.txt 2>&1
  echo "gsm $i $(date -u +%H:%MZ) $(grep -a -i accuracy $2/gsm_eval_$i.txt | tail -n 1)" >> $2/log
done
