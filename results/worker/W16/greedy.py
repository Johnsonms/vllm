"""Greedy check: 32 single prompts (one request each) + 256 prompts in ONE request, 256 tokens, temperature 0.
The 256-prompt request is sent twice; the first one after server start is discarded. usage: greedy.py PORT OUTDIR"""
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
M = "Qwen/Qwen3.8-Flash-Next-FP8"
topics = ["a river", "prime numbers", "a steam engine", "photosynthesis", "a binary search", "the moon", "a hash table", "glaciers",
          "compound interest", "a suspension bridge", "the water cycle", "a neural network", "volcanoes", "a sorting algorithm", "tides", "vaccines"]
kinds = ["Explain {} in three sentences.", "Write a short poem about {}.", "List five facts about {}.", "Describe {} to a ten-year-old."]
prompts = [f"{kinds[i % 4].format(topics[(i // 4) % 16])} (variant {i // 64 + 1}; then compute {17 + i} * {3 + i % 7} step by step.)" for i in range(256)]
def call(p):
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(dict(model=M, prompt=p, max_tokens=256, temperature=0, seed=0)).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return [c["text"] for c in sorted(r["choices"], key=lambda c: c["index"])]
single = [call(p)[0] for p in prompts[:32]]
call(prompts)  # discarded
batched = call(prompts)
json.dump(dict(single=single, batched=batched), open(f"{out}/greedy.json", "w"))
print("greedy done", len(single), len(batched))
