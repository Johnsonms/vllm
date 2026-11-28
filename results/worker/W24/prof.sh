#!/bin/bash
# usage: prof.sh ARM GPU PORT C  -> decode profile at concurrency C (skip 24 steps, record 32) + unprofiled TPOT
source ~/kopt-env.sh; ARM=$1; GPU=$2; PORT=$3; C=$4; TREE=$HOME/vllm-kopt-w16a; OUT=/scratch/kopt-w24/prof_${ARM}_c$C; rm -rf $OUT; mkdir -p $OUT/traces; cd $TREE
export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONPATH=$TREE; M=Qwen/Qwen3-8B-FP8
if [ $ARM = mx ]; then export KOPT_MX_MLP=1 KOPT_MX_SNAPSHOT=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/*) KOPT_MX_MIN_M=16; fi
PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
CUDA_VISIBLE_DEVICES=$GPU setsid python -m vllm.entrypoints.cli.main serve $M --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.9 --no-enable-prefix-caching --max-num-seqs 256 --profiler-config "$PCFG" > $OUT/serve.log 2>&1 &
SPID=$!
for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 10; done
B="python -m vllm.entrypoints.cli.main bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random --random-range-ratio 0 --ignore-eos --seed 0 --random-input-len 512"
for K in 1 2 3; do $B --random-output-len 256 --num-prompts $((5*C)) --max-concurrency $C --num-warmups $C --percentile-metrics ttft,tpot,itl --metric-percentiles 50 --save-result --result-dir $OUT --result-filename tpot_$K.json > $OUT/tpot_$K.txt 2>&1; done
$B --random-output-len 128 --num-prompts $C --max-concurrency $C --num-warmups $C --profile > $OUT/prof.txt 2>&1
sleep 30; kill -INT -- -$SPID; sleep 15; echo "prof $ARM c$C done" >> /scratch/kopt-w24/status
