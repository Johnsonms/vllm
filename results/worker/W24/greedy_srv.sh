#!/bin/bash
# usage: greedy_srv.sh NAME ARM MIN_M GPU PORT : start server, run greedy.py (64 single prompts, 2 repeats), stop
source ~/kopt-env.sh; NAME=$1; ARM=$2; export KOPT_MX_MIN_M=$3; GPU=$4; PORT=$5; OUT=/scratch/kopt-w24/g_$NAME
setsid /scratch/kopt-w23/serve.sh $ARM $GPU $PORT $OUT & SPID=$!
for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 10; done
python /scratch/kopt-w24/greedy.py $PORT $OUT/greedy.json > $OUT/greedy.txt 2>&1
kill -INT -- -$SPID; sleep 15; echo "greedy $NAME done" >> /scratch/kopt-w24/status
