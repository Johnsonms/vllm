"""Run tests/evals/gsm8k/gsm8k_eval.py (from the cwd tree) and also dump per-question correctness.
No vLLM edit: wraps _score_gsm8k. usage: python gsm8k_perq.py PERQ.json <gsm8k_eval args...>"""
import importlib.util, json, sys
out = sys.argv[1]
spec = importlib.util.spec_from_file_location("gsm8k_eval", "tests/evals/gsm8k/gsm8k_eval.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig = m._score_gsm8k
def wrapped(states, output_tokens, labels, *a, **k):
    preds = [m.get_answer_value(s) for s in states]
    json.dump({"correct": [int(p == l) for p, l in zip(preds, labels)], "preds": preds, "labels": list(labels),
               "invalid": [int(p == m.INVALID) for p in preds], "output_tokens": list(output_tokens)}, open(out, "w"))
    return orig(states, output_tokens, labels, *a, **k)
m._score_gsm8k = wrapped
sys.argv = ["gsm8k_eval.py"] + sys.argv[2:]
m.main()
