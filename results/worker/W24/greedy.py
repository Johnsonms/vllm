"""64 single prompts, greedy, 128 tokens, one request at a time (M = 1 decode), each prompt twice (run-to-run floor)."""
import json, sys, urllib.request
port, outp = sys.argv[1], sys.argv[2]
topics = ["the ocean", "prime numbers", "a rainy city", "the history of bread", "black holes", "learning to swim", "a lost dog", "compilers"]
forms = ["Write a short paragraph about {}.", "Explain {} to a ten year old.", "List five facts about {}.", "Write a four line poem about {}.",
         "What is surprising about {}?", "Give step by step advice related to {}.", "Tell a very short story involving {}.", "Summarize what people get wrong about {}."]
prompts = [f.format(t) for t in topics for f in forms]
def gen(p):
    body = json.dumps(dict(model="Qwen/Qwen3-8B-FP8", prompt=p, max_tokens=128, temperature=0, logprobs=1)).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/completions", body, {"Content-Type": "application/json"})))
    return r["choices"][0]["logprobs"]["tokens"]
res = [[gen(p) for p in prompts] for _ in range(2)]
json.dump(dict(prompts=prompts, runs=res), open(outp, "w"))
print("self-identical", sum(a == b for a, b in zip(*res)), "of", len(prompts))
