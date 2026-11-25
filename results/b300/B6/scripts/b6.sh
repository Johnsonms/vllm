#!/bin/bash
# B6: c2/c4/c8 decode profile, Qwen3.6-35B-A3B-FP8 TP1 real weights, GPU 0, B1 profiling method.
# Arms: base (e66bede56b) and pr (B3 arm: 81159c69d6 table + (10, 3)); only skinny_gemm_plans.py differs.
# Per arm and C: 1 unprofiled TPOT run (512/512, as B3), then 1 profiled run (num-prompts C,
# 512/128; skip 24 engine steps after /start_profile, record 32).
source ~/kopt-env.sh
export VLLM_DISABLE_COMPILE_CACHE=1
M=Qwen/Qwen3.6-35B-A3B-FP8; PORT=8861; ROOT=~/b300-work/B6; ORDER=${ORDER:-"base pr"}
TREE=~/vllm-kopt; F=vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm_plans.py
cd $TREE
SPID=
cleanup() { local j; kill -INT $SPID 2>/dev/null; for j in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; sleep 5; }
stepback() { echo "$(date +%H:%M:%S) STEPPED_BACK $1" >> $ROOT/status; [ -n "$SPID" ] && cleanup; cp ~/b300-work/B3/arms/base.py $TREE/$F; exit 2; }
others() { ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$1"; }
for ARM in $ORDER; do
  OUT=$ROOT/$ARM; mkdir -p $OUT; others "before $ARM"
  cp ~/b300-work/B3/arms/$ARM.py $TREE/$F
  python - > $OUT/serve.log 2>&1 <<'PY'
import hashlib, vllm, vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans as p
print("CHECK vllm from", vllm.__file__)
print("CHECK plans file", p.__file__, "sha1", hashlib.sha1(open(p.__file__, "rb").read()).hexdigest()[:12])
print("CHECK (10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY:", (10, 3) in p.SKINNY_GEMM_PLANS_BY_CAPABILITY)
PY
  PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
  CUDA_VISIBLE_DEVICES=0 vllm serve $M --port $PORT --language-model-only --max-num-seqs 256 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching --profiler-config "$PCFG" >> $OUT/serve.log 2>&1 &
  SPID=$!
  for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
  if ! curl -sf localhost:$PORT/health >/dev/null; then echo "$(date +%H:%M:%S) SERVER_FAILED $ARM" >> $ROOT/status; cleanup; cp ~/b300-work/B3/arms/base.py $TREE/$F; exit 1; fi
  for C in 2 4 8; do
    others "$ARM c$C"
    N=$(( 4*C > 16 ? 4*C : 16 ))
    vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $N \
      --max-concurrency $C --num-warmups $C --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
      --save-result --result-dir $OUT --result-filename tpot_c$C.json > $OUT/tpot_c$C.txt 2>&1
    others "$ARM prof c$C"; mkdir -p $OUT/traces
    vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 128 --random-range-ratio 0 --ignore-eos --num-prompts $C \
      --max-concurrency $C --num-warmups $C --seed 0 --profile \
      --save-result --result-dir $OUT --result-filename prof_c$C.json > $OUT/prof_c$C.txt 2>&1
    sleep 30; mkdir -p $OUT/traces_c$C; for f in $OUT/traces/*; do mv "$f" $OUT/traces_c$C/; done
  done
  cleanup; SPID=
  echo "$(date +%H:%M:%S) done $ARM" >> $ROOT/status
done
cp ~/b300-work/B3/arms/base.py $TREE/$F
echo DONE >> $ROOT/status
