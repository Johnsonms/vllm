#!/bin/bash
# greedy token check: 4 prompts, both servers; usage tokens.sh PORT_A PORT_B ROOT
M=Qwen/Qwen3.6-35B-A3B-FP8; i=0
for PR in "Explain why the sky is blue in three sentences." "Write a Python function that returns the n-th Fibonacci number." "List the first ten prime numbers and their sum." "Translate to French: The weather is nice today, let us walk to the market."; do
  i=$((i+1)); for X in "a $1" "b $2"; do set -- $1 $2 $3; A=${X% *}; P=${X#* }
    curl -s localhost:$P/v1/completions -H 'Content-Type: application/json' -d "$(python3 -c 'import json,sys;print(json.dumps(dict(model=sys.argv[1],prompt=sys.argv[2],max_tokens=256,temperature=0,seed=0)))' $M "$PR")" | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])' > $3/tok_${A}_$i.txt
  done
  cmp -s $3/tok_a_$i.txt $3/tok_b_$i.txt && echo "prompt $i IDENTICAL $(wc -c < $3/tok_a_$i.txt) bytes" >> $3/tokens.txt || echo "prompt $i DIFFER" >> $3/tokens.txt
done
