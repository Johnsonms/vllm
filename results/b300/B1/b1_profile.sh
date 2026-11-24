#!/bin/bash
# B1 step 4: decode profile, Qwen3.6-35B-A3B-FP8, TP1, dummy weights, GPU 0.
# Per concurrency C in {1, 8}: 2 unprofiled TPOT runs, then 1 profiled run (num-prompts C).
# Profiler window: skip 24 engine steps after /start_profile (prefill done), then record 32 steps.
source ~/kopt-env.sh
~/b300-work/gpu_others.sh || { echo STEPPED_BACK_BEFORE_START > ~/b300-work/B1_profile.status; exit 2; }
export VLLM_DISABLE_COMPILE_CACHE=1
cd ~/vllm-kopt   # launch from the tree being measured
M=Qwen/Qwen3.6-35B-A3B-FP8; PORT=8831; ROOT=~/b300-work/B1_profile; mkdir -p $ROOT
python -c "import vllm; print('vllm from', vllm.__file__)" > $ROOT/serve.log
PCFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$ROOT/traces\",\"delay_iterations\":24,\"max_iterations\":32,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
CUDA_VISIBLE_DEVICES=0 vllm serve $M --port $PORT --load-format dummy --language-model-only \
  --max-model-len 4096 --max-num-seqs 64 --no-enable-prefix-caching \
  --profiler-config "$PCFG" >> $ROOT/serve.log 2>&1 &
SPID=$!
cleanup() { kill -INT $SPID 2>/dev/null; for i in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; }
for i in $(seq 1 120); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 10; done
if ! curl -sf localhost:$PORT/health >/dev/null; then echo SERVER_FAILED > $ROOT/FAILED; cleanup; exit 1; fi
bench() {  # $1=C $2=num_prompts $3=name $4=extra
  if ! ~/b300-work/gpu_others.sh >> $ROOT/stepback.log; then cleanup; echo STEPPED_BACK > $ROOT/status; exit 2; fi
  vllm bench serve --backend openai --endpoint /v1/completions --model $M --port $PORT --dataset-name random \
    --random-input-len 512 --random-output-len 128 --random-range-ratio 0 --ignore-eos --num-prompts $2 \
    --max-concurrency $1 --num-warmups $1 --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
    --save-result --result-dir $ROOT --result-filename $3.json $4 > $ROOT/$3.txt 2>&1
}
for C in 1 8; do
  bench $C $((C*8)) tpot_c${C}_r1
  bench $C $((C*8)) tpot_c${C}_r2
  mkdir -p $ROOT/traces; before=$(ls $ROOT/traces | wc -l)
  bench $C $C prof_c$C --profile
  sleep 30   # trace flush
  mkdir -p $ROOT/traces_c$C; for f in $ROOT/traces/*; do mv "$f" $ROOT/traces_c$C/; done
done
cleanup
echo DONE > $ROOT/status
