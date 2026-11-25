#!/bin/bash
# B7 timeline check: arms T (e66bede56b + (10,3) table, aux priority 0) and P (same + aux priority -1).
# Same tree (~/vllm-kopt, local edits); P differs only by KOPT_B7_AUX_PRIORITY=-1. B6 method at c4, c8.
source ~/kopt-env.sh
export VLLM_DISABLE_COMPILE_CACHE=1
M=Qwen/Qwen3.6-35B-A3B-FP8; PORT=8871; ROOT=~/b300-work/B7/prof; ORDER=${ORDER:-"T P"}; mkdir -p $ROOT
cd ~/vllm-kopt
SPID=
cleanup() { local j; kill -INT $SPID 2>/dev/null; for j in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; sleep 5; }
stepback() { echo "$(date +%H:%M:%S) STEPPED_BACK $1" >> $ROOT/status; [ -n "$SPID" ] && cleanup; exit 2; }
others() { ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$1"; }
for ARM in $ORDER; do
  OUT=$ROOT/$ARM; mkdir -p $OUT; others "before $ARM"
  [ $ARM = P ] && PR=-1 || PR=0
  KOPT_B7_AUX_PRIORITY=$PR python - > $OUT/serve.log 2>&1 <<'PY'
import vllm, vllm.utils.torch_utils as t, vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans as p
print("CHECK vllm from", vllm.__file__)
print("CHECK (10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY:", (10, 3) in p.SKINNY_GEMM_PLANS_BY_CAPABILITY)
print("CHECK torch_utils has KOPT_B7 edit:", "KOPT_B7_AUX_PRIORITY" in open(t.__file__).read(), t.__file__)
PY
  PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
  KOPT_B7_AUX_PRIORITY=$PR CUDA_VISIBLE_DEVICES=0 vllm serve $M --port $PORT --language-model-only --max-num-seqs 256 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --no-enable-prefix-caching --profiler-config "$PCFG" >> $OUT/serve.log 2>&1 &
  SPID=$!
  for j in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
  if ! curl -sf localhost:$PORT/health >/dev/null; then echo "$(date +%H:%M:%S) SERVER_FAILED $ARM" >> $ROOT/status; cleanup; exit 1; fi
  for C in 4 8; do
    others "$ARM c$C"; mkdir -p $OUT/traces
    vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
      --random-input-len 512 --random-output-len 128 --random-range-ratio 0 --ignore-eos --num-prompts $C \
      --max-concurrency $C --num-warmups $C --seed 0 --profile \
      --save-result --result-dir $OUT --result-filename prof_c$C.json > $OUT/prof_c$C.txt 2>&1
    sleep 30; mkdir -p $OUT/traces_c$C; for f in $OUT/traces/*; do mv "$f" $OUT/traces_c$C/; done
  done
  cleanup; SPID=
  echo "$(date +%H:%M:%S) done $ARM" >> $ROOT/status
done
echo DONE >> $ROOT/status
