"""Run tests/evals/gsm8k/gsm8k_eval.py unchanged, but also dump per-question prediction / label / text. usage: GSM_DUMP=out.json gsm_dump.py <eval args>"""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("gsm8k_eval", os.path.expanduser("~/vllm-kopt/tests/evals/gsm8k/gsm8k_eval.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig = m._score_gsm8k
def wrapped(states, output_tokens, labels, *a, **k):
    preds = [m.get_answer_value(s) for s in states]
    json.dump(dict(preds=[int(p) for p in preds], labels=[int(l) for l in labels], tokens=list(output_tokens), states=states), open(os.environ["GSM_DUMP"], "w"))
    return orig(states, output_tokens, labels, *a, **k)
m._score_gsm8k = wrapped
m.main()
