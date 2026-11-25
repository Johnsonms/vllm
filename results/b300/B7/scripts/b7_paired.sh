#!/bin/bash
# B7 paired A/B (W12c design): T (table, aux prio 0) and P (table, aux prio -1) servers up at once on two GPUs.
# Per c in 1 2 4 8 16: 6 rounds, one burst per arm per round (5*c prompts, 512/512), order flipped each round.
# Pass 1: T on GPU 0, P on GPU 1. Pass 2: swapped. Per pass: greedy-token check (4 prompts) and one GSM8K per arm.
source ~/kopt-env.sh
export VLLM_DISABLE_COMPILE_CACHE=1
M=Qwen/Qwen3.6-35B-A3B-FP8; ROOT=~/b300-work/B7/paired; mkdir -p $ROOT; PASSES=${PASSES:-"1 2"}
cd ~/vllm-kopt
declare -A PORT=([T]=8881 [P]=8882) PRIO=([T]=0 [P]=-1)
cleanup() { local j; kill -INT ${SP[T]} ${SP[P]} 2>/dev/null; for j in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; sleep 5; }
stepback() { echo "$(date +%H:%M:%S) STEPPED_BACK $1" >> $ROOT/status; cleanup; exit 2; }
others() { ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$1"; }
declare -A SP
for PASS in $PASSES; do
  if [ $PASS = 1 ]; then declare -A GPU=([T]=0 [P]=1); else declare -A GPU=([T]=1 [P]=0); fi
  others "pass $PASS start"
  for A in T P; do
    D=$ROOT/pass$PASS/$A; mkdir -p $D
    KOPT_B7_AUX_PRIORITY=${PRIO[$A]} python - > $D/serve.log 2>&1 <<'PY'
import vllm, vllm.utils.torch_utils as t, vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans as p
print("CHECK vllm from", vllm.__file__)
print("CHECK (10, 3) in SKINNY_GEMM_PLANS_BY_CAPABILITY:", (10, 3) in p.SKINNY_GEMM_PLANS_BY_CAPABILITY)
print("CHECK torch_utils has KOPT_B7 edit:", "KOPT_B7_AUX_PRIORITY" in open(t.__file__).read(), t.__file__)
PY
    echo "CHECK arm=$A gpu=${GPU[$A]} prio=${PRIO[$A]}" >> $D/serve.log
    KOPT_B7_AUX_PRIORITY=${PRIO[$A]} CUDA_VISIBLE_DEVICES=${GPU[$A]} vllm serve $M --port ${PORT[$A]} --language-model-only \
      --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching >> $D/serve.log 2>&1 &
    SP[$A]=$!
  done
  for A in T P; do
    for j in $(seq 1 150); do curl -sf localhost:${PORT[$A]}/health >/dev/null && break; sleep 10; done
    curl -sf localhost:${PORT[$A]}/health >/dev/null || { echo "$(date +%H:%M:%S) SERVER_FAILED pass$PASS $A" >> $ROOT/status; cleanup; exit 1; }
  done
  # greedy-token identity check
  for A in T P; do
    python3 - ${PORT[$A]} > $ROOT/pass$PASS/greedy_$A.json <<'PY'
import json, sys, urllib.request
port = sys.argv[1]; out = []
for p in ["The capital of France is", "def fibonacci(n):", "Explain why the sky is blue in one paragraph.", "1, 1, 2, 3, 5, 8,"]:
    req = {"model": "Qwen/Qwen3.6-35B-A3B-FP8", "prompt": p, "max_tokens": 64, "temperature": 0, "logprobs": 1}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})))
    c = r["choices"][0]; out.append({"prompt": p, "text": c["text"], "tokens": c["logprobs"]["tokens"]})
print(json.dumps(out))
PY
  done
  for C in 1 2 4 8 16; do
    for R in 1 2 3 4 5 6; do
      if [ $((R % 2)) = 1 ]; then ORD="T P"; else ORD="P T"; fi
      for A in $ORD; do
        others "pass$PASS c$C r$R $A"
        vllm bench serve --backend openai --endpoint /v1/completions --model $M --port ${PORT[$A]} --dataset-name random \
          --random-input-len 512 --random-output-len 512 --random-range-ratio 0 --ignore-eos --num-prompts $((5*C)) \
          --max-concurrency $C --num-warmups $C --seed $R --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
          --save-result --result-dir $ROOT/pass$PASS/$A --result-filename c${C}_r$R.json > $ROOT/pass$PASS/$A/c${C}_r$R.txt 2>&1
      done
    done
    echo "$(date +%H:%M:%S) pass$PASS c$C done" >> $ROOT/status
  done
  for A in T P; do
    others "pass$PASS gsm8k $A"
    python tests/evals/gsm8k/gsm8k_eval.py --port ${PORT[$A]} --use-chat-completions \
      --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64 \
      --save-results $ROOT/pass$PASS/$A/gsm8k.json > $ROOT/pass$PASS/$A/gsm8k.txt 2>&1
  done
  cleanup
  echo "$(date +%H:%M:%S) pass$PASS done" >> $ROOT/status
done
echo DONE >> $ROOT/status
