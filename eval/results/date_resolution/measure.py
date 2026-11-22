"""LoCoMo DEV only: evidence recall, declines, and whether the reader's answer matches a time question's gold."""
import contextlib, io, json, re, sys
sys.path[:0] = [".", "eval", "eval/benchmarks"]
import numpy as np
import run_eval, locomo
env = run_eval.Env("st:BAAI/bge-small-en-v1.5")
from backend.api.deps import get_memory_manager, reset
reset()
with contextlib.redirect_stdout(io.StringIO()):
    mm = get_memory_manager()

MONTHS = "january february march april may june july august september october november december".split()
def toks(s):
    s = str(s).lower().replace(",", " ").replace(".", " ")
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s); s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    return [t for t in s.split() if t not in ("the", "on", "in", "of", "a", "at")]
def matches(answer, gold):
    g, a = toks(gold), set(toks(answer or ""))
    key = [t for t in g if t.isdigit() or t in MONTHS or t in ("week", "weekend", "before", "after") or t.endswith("day")]
    return bool(key) and all(t in a for t in key)

rows = []
for sample in locomo.split_samples(json.load(open(sys.argv[1])), "dev"):
    user, m2d = locomo.ingest(mm, sample)
    for qa in sample["qa"]:
        with contextlib.redirect_stdout(io.StringIO()):
            out = mm.retrieve(user, qa["question"], limit=10)
        served = [d for m in out["memories"] for d in m2d[m["id"]]]
        ev = locomo.norm_evidence(qa.get("evidence")) & set(d for d, _, _ in locomo.turns_of(sample))
        top = out["memories"][0] if out["memories"] else {}
        rows.append({"cat": qa["category"], "declined": out["status"] in ("ABSTAINED", "NO_MATCH"),
                     "recall10": (len(set(served) & ev) / len(ev)) if ev else None,
                     "answer": top.get("answer"), "gold": qa.get("answer"),
                     "has_dates": bool((top.get("metadata") or {}).get("dates")) if isinstance(top.get("metadata"), dict) else False})
ans = [r for r in rows if r["cat"] != 5]; adv = [r for r in rows if r["cat"] == 5]
res = {"recall@10": round(float(np.mean([r["recall10"] for r in ans if r["recall10"] is not None])), 4),
       "recall@10_by_cat": {c: round(float(np.mean([r["recall10"] for r in ans if r["cat"] == c and r["recall10"] is not None])), 4) for c in (1, 2, 3, 4)},
       "adversarial_declined": round(float(np.mean([r["declined"] for r in adv])), 4),
       "answerable_wrongly_declined": round(float(np.mean([r["declined"] for r in ans])), 4),
       "temporal_reader_answer_matches_gold": f'{sum(matches(r["answer"], r["gold"]) for r in ans if r["cat"] == 2)}/{sum(1 for r in ans if r["cat"] == 2)}'}
json.dump({"summary": res, "rows": rows}, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res))
