#!/bin/bash
# usage: paired.sh EXP GPU_A GPU_B PORT_BASE
# Paired A/B, both servers up at once, one tree (flags select the arm). 2 passes x 6 rounds,
# GPUs swapped in pass 2; per round one burst per arm, order flipped each round.
# TTFT: 4096 in / 1 out, c1/c4/c16. TPOT: 512/512, c1/c8/c32/c128. GSM8K once per arm (pass 1).
EXP=$1; G0=$2; G1=$3; PB=$4
source ~/kopt-env.sh; source ~/kopt-work/killtree.sh
T=~/kopt-work/tree; M=${MODEL:-Qwen/Qwen3-30B-A3B}; ROOT=~/kopt-work/${TASK:-H2}/e2e/${TAG:-$EXP}; mkdir -p $ROOT
case $EXP in
  def_vs_pr)    BASE_ENV="";                      BASE_ARGS="";                      PR_ENV="KOPT_QUACK_TOPK=1 KOPT_QUACK_MOE=1"; PR_ARGS="--moe-backend triton" ;;
  tri_vs_pr)    BASE_ENV="";                      BASE_ARGS="--moe-backend triton";  PR_ENV="KOPT_QUACK_TOPK=1 KOPT_QUACK_MOE=1"; PR_ARGS="--moe-backend triton" ;;
  def_vs_defp1) BASE_ENV="";                      BASE_ARGS="";                      PR_ENV="KOPT_QUACK_TOPK=1";                  PR_ARGS="" ;;
  swiglu)       BASE_ENV="";                      BASE_ARGS="";                      PR_ENV="KOPT_QUACK_SWIGLU=1 KOPT_SWIGLU_SMALL_M=128 KOPT_SWIGLU_TABLE=$HOME/kopt-work/H3/sm90_table.json"; PR_ARGS="" ;;
  *) echo "unknown exp"; exit 1 ;;
esac
declare -A PORT=([base]=$PB [pr]=$((PB+1))) ENVS=([base]="$BASE_ENV" [pr]="$PR_ENV") ARGS=([base]="$BASE_ARGS" [pr]="$PR_ARGS")
declare -A SP
log() { echo "$(date -u +%H:%M:%SZ) $*" >> $ROOT/status; }
stopcheck() { if [ -f ~/kopt-work/STOP ]; then log "STOP file present, aborting"; cleanup; exit 3; fi; }
cleanup() { stop_server ${SP[base]}; stop_server ${SP[pr]}; sleep 3; }
TTFT_C=${TTFT_C:-"1 4 16"}; TPOT_C=${TPOT_C:-"1 8 32 128"}
for PASS in ${PASSES:-1 2}; do
  if [ $PASS = 1 ]; then declare -A GPU=([base]=$G0 [pr]=$G1); else declare -A GPU=([base]=$G1 [pr]=$G0); fi
  stopcheck
  for A in base pr; do
    D=$ROOT/pass$PASS/$A; mkdir -p $D
    (cd $T && PYTHONPATH=$T python -c "import vllm,subprocess;print('CHECK vllm from',vllm.__file__);print('CHECK tree',subprocess.run(['git','diff','--stat'],capture_output=True,text=True).stdout.strip().splitlines()[-1:])" > $D/serve.log 2>&1)
    echo "CHECK arm=$A gpu=${GPU[$A]} env='${ENVS[$A]}' args='${ARGS[$A]}'" >> $D/serve.log
    (cd $T && exec env PYTHONPATH=$T ${ENVS[$A]} CUDA_VISIBLE_DEVICES=${GPU[$A]} vllm serve $M --port ${PORT[$A]} \
      --max-num-seqs 256 --max-model-len 8192 --gpu-memory-utilization 0.9 --no-enable-prefix-caching ${ARGS[$A]} ${EXTRA_ARGS:-} >> $D/serve.log 2>&1) &
    SP[$A]=$!
  done
  for A in base pr; do
    for j in $(seq 1 120); do curl -sf localhost:${PORT[$A]}/health >/dev/null && break; kill -0 ${SP[$A]} 2>/dev/null || break; sleep 10; done
    curl -sf localhost:${PORT[$A]}/health >/dev/null || { log "SERVER_FAILED pass$PASS $A"; cleanup; exit 1; }
  done
  log "pass$PASS servers up (base GPU ${GPU[base]}, pr GPU ${GPU[pr]})"
  for A in base pr; do
    python3 - ${PORT[$A]} $M > $ROOT/pass$PASS/greedy_$A.json <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]; out = []
for p in ["The capital of France is", "def fibonacci(n):", "Explain why the sky is blue in one paragraph.", "1, 1, 2, 3, 5, 8,"]:
    req = {"model": model, "prompt": p, "max_tokens": 64, "temperature": 0, "logprobs": 1}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})))
    c = r["choices"][0]; out.append({"prompt": p, "text": c["text"], "tokens": c["logprobs"]["tokens"]})
print(json.dumps(out))
PY
  done
  bench() { # kind c r arm
    local IN OUT; if [ $1 = ttft ]; then IN=4096; OUT=1; else IN=512; OUT=512; fi
    vllm bench serve --backend openai --endpoint /v1/completions --model $M --port ${PORT[$4]} --dataset-name random \
      --random-input-len $IN --random-output-len $OUT --random-range-ratio 0 --ignore-eos --num-prompts $((5*$2)) \
      --max-concurrency $2 --num-warmups $2 --seed $3 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
      --save-result --result-dir $ROOT/pass$PASS/$4 --result-filename $1_c$2_r$3.json > $ROOT/pass$PASS/$4/$1_c$2_r$3.txt 2>&1
  }
  for KIND in ttft tpot; do
    if [ $KIND = ttft ]; then CS=$TTFT_C; else CS=$TPOT_C; fi
    for C in $CS; do
      for R in 1 2 3 4 5 6; do
        if [ $((R % 2)) = 1 ]; then ORD="base pr"; else ORD="pr base"; fi
        for A in $ORD; do stopcheck; bench $KIND $C $R $A; done
      done
      log "pass$PASS $KIND c$C done"
    done
  done
  if [ $PASS = 1 ] && [ "${GSM:-1}" = 1 ]; then
    for A in base pr; do
      stopcheck
      (cd $T && python tests/evals/gsm8k/gsm8k_eval.py --port ${PORT[$A]} --use-chat-completions \
        --chat-template-kwargs '{"enable_thinking": false}' --max-tokens 1024 --temperature 0 --max-concurrency 64 \
        --save-results $ROOT/pass1/$A/gsm8k.json > $ROOT/pass1/$A/gsm8k.txt 2>&1)
      log "gsm8k $A $(grep -a Accuracy $ROOT/pass1/$A/gsm8k.txt)"
    done
  fi
  for A in base pr; do echo "AOT_loads $(grep -ac 'Directly load AOT' $ROOT/pass$PASS/$A/serve.log)" >> $ROOT/pass$PASS/$A/checks.txt; done
  cleanup
  log "pass$PASS done"
done
log DONE
