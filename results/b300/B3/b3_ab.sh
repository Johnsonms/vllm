#!/bin/bash
# B3: E2E A/B on B300, Qwen3.6-35B-A3B-FP8 TP1 real weights. Arms differ only in
# vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py (base = e66bede56b,
# pr = 81159c69d6 + "(10, 3): SM100_SKINNY_GEMM_PLANS,"). Order base, pr, base, pr.
# usage: b3_ab.sh TP GPUS "C list" [order]
source ~/kopt-env.sh
export VLLM_DISABLE_COMPILE_CACHE=1
TP=${1:-1}; GPUS=${2:-0}; CS=${3:-"1 2 4 8"}; ORDER=${4:-"base pr base pr"}
M=Qwen/Qwen3.6-35B-A3B-FP8; PORT=8851
TREE=~/vllm-kopt; F=vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py
ROOT=~/b300-work/B3/tp$TP; mkdir -p $ROOT
cd $TREE   # launch from the tree being timed
stepback() { echo "$(date +%H:%M:%S) STEPPED_BACK $1" >> $ROOT/status; [ -n "$SPID" ] && cleanup; cp ~/b300-work/B3/arms/base.py $TREE/$F; exit 2; }
cleanup() { local j; kill -INT $SPID 2>/dev/null; for j in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; sleep 5; }
i=0
for ARM in $ORDER; do
  i=$((i+1)); OUT=$ROOT/r${i}_$ARM; mkdir -p $OUT
  ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "before $ARM r$i"
  cp ~/b300-work/B3/arms/$ARM.py $TREE/$F
  python - > $OUT/serve.log 2>&1 <<'EOF'
import hashlib, vllm, vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans as p
print("CHECK vllm from", vllm.__file__)
print("CHECK plans file", p.__file__, "sha1", hashlib.sha1(open(p.__file__, "rb").read()).hexdigest()[:12])
print("CHECK (10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY:", (10, 3) in p.SKINNY_GEMM_PLANS_BY_CAPABILITY)
print("CHECK keys", sorted(k for k in p.SKINNY_GEMM_PLANS_BY_CAPABILITY if k))
print("CHECK plan(256, 2048) on this GPU:", p.get_skinny_gemm_plan(256, 2048) is not None)
EOF
  CUDA_VISIBLE_DEVICES=$GPUS vllm serve $M --port $PORT -tp $TP --language-model-only --max-num-seqs 256 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching >> $OUT/serve.log 2>&1 &
  SPID=$!
  for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
  if ! curl -sf localhost:$PORT/health >/dev/null; then echo "$(date +%H:%M:%S) SERVER_FAILED $ARM r$i" >> $ROOT/status; cleanup; cp ~/b300-work/B3/arms/base.py $TREE/$F; exit 1; fi
  for C in $CS; do
    ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$ARM r$i c$C"
    N=$(( 4*C > 16 ? 4*C : 16 ))
    vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $N \
      --max-concurrency $C --num-warmups $C --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
      --save-result --result-dir $OUT --result-filename c$C.json > $OUT/c$C.txt 2>&1
  done
  ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$ARM r$i gsm8k"
  python tests/evals/gsm8k/gsm8k_eval.py --port $PORT --use-chat-completions \
    --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64 \
    --save-results $OUT/gsm8k.json > $OUT/gsm8k.txt 2>&1
  cleanup; SPID=
  echo "$(date +%H:%M:%S) done $ARM r$i" >> $ROOT/status
done
cp ~/b300-work/B3/arms/base.py $TREE/$F
echo DONE >> $ROOT/status
