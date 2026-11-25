#!/bin/bash
# B8 paired A/B: base (fb159244e6) vs pr (c5ef00f633 = #69778) servers up at once on two GPUs, one tree each
# (git worktrees ~/b8/{base,pr}; PYTHONPATH and cwd set per server). Per c in 1 2 4 8: 6 rounds, one burst per arm
# per round (5*c prompts, 512/512), order flipped each round. Pass 1: base on GPU 0, pr on GPU 1; pass 2 swapped.
# Per pass: greedy-token check (4 prompts) and 4 GSM8K runs per arm, interleaved, with per-question dumps.
source ~/kopt-env.sh
export VLLM_DISABLE_COMPILE_CACHE=1
M=Qwen/Qwen3.6-35B-A3B-FP8; ROOT=~/b300-work/B8/paired; mkdir -p $ROOT; PASSES=${PASSES:-"1 2"}
declare -A PORT=([base]=8891 [pr]=8892) TREE=([base]=$HOME/b8/base [pr]=$HOME/b8/pr)
declare -A SP
cleanup() { local j; kill -INT ${SP[base]} ${SP[pr]} 2>/dev/null; for j in $(seq 1 30); do pgrep -u $USER -f '[V]LLM::Worker|[V]LLM::EngineCore' >/dev/null || break; sleep 2; done
  pgrep -u $USER -f '[V]LLM::|[v]llm serve Qwen' | xargs -r kill -9; sleep 5; }
stepback() { echo "$(date +%H:%M:%S) STEPPED_BACK $1" >> $ROOT/status; cleanup; exit 2; }
others() { ~/b300-work/gpu_others.sh >> $ROOT/stepback.log || stepback "$1"; }
for PASS in $PASSES; do
  if [ $PASS = 1 ]; then declare -A GPU=([base]=0 [pr]=1); else declare -A GPU=([base]=1 [pr]=0); fi
  others "pass $PASS start"
  for A in base pr; do
    D=$ROOT/pass$PASS/$A; mkdir -p $D
    (cd ${TREE[$A]} && PYTHONPATH=${TREE[$A]} python - > $D/serve.log 2>&1 <<'PY'
import subprocess, vllm, vllm.model_executor.layers.quantization.fp8 as f
print("CHECK vllm from", vllm.__file__)
print("CHECK tree HEAD", subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()[:70])
print("CHECK fp8.py has expose_input_quant_key:", "expose_input_quant_key" in open(f.__file__).read())
PY
    )
    echo "CHECK arm=$A gpu=${GPU[$A]}" >> $D/serve.log
    (cd ${TREE[$A]} && PYTHONPATH=${TREE[$A]} CUDA_VISIBLE_DEVICES=${GPU[$A]} exec vllm serve $M --port ${PORT[$A]} --language-model-only \
      --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching >> $D/serve.log 2>&1) &
    SP[$A]=$!
  done
  for A in base pr; do
    for j in $(seq 1 150); do curl -sf localhost:${PORT[$A]}/health >/dev/null && break; sleep 10; done
    curl -sf localhost:${PORT[$A]}/health >/dev/null || { echo "$(date +%H:%M:%S) SERVER_FAILED pass$PASS $A" >> $ROOT/status; cleanup; exit 1; }
  done
  for A in base pr; do
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
  for C in 1 2 4 8; do
    for R in 1 2 3 4 5 6; do
      if [ $((R % 2)) = 1 ]; then ORD="base pr"; else ORD="pr base"; fi
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
  for G in 1 2 3 4; do
    if [ $((G % 2)) = 1 ]; then ORD="base pr"; else ORD="pr base"; fi
    for A in $ORD; do
      others "pass$PASS gsm8k $G $A"
      (cd ${TREE[$A]} && python ~/b300-work/B8/gsm8k_perq.py $ROOT/pass$PASS/$A/gsm8k_g${G}_perq.json --port ${PORT[$A]} --use-chat-completions \
        --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64 \
        --save-results $ROOT/pass$PASS/$A/gsm8k_g$G.json > $ROOT/pass$PASS/$A/gsm8k_g$G.txt 2>&1)
    done
  done
  echo "$(date +%H:%M:%S) pass$PASS gsm8k done" >> $ROOT/status
  cleanup
  echo "$(date +%H:%M:%S) pass$PASS done" >> $ROOT/status
done
echo DONE >> $ROOT/status
