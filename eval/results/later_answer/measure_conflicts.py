"""LoCoMo DEV: on questions where recall found a later, different answer (answer.later_answer), which one matches
the gold: the confident one (answer.text) or the later one? Development data only."""
import collections, contextlib, io, json, re, sys
sys.path[:0] = [".", "eval", "eval/benchmarks"]
import run_eval, locomo
env = run_eval.Env("st:BAAI/bge-small-en-v1.5")
from backend.api.deps import get_memory_manager, reset
reset()
with contextlib.redirect_stdout(io.StringIO()):
    mm = get_memory_manager()

def f1(a, g):
    tk = lambda x: [t for t in re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split()
                    if t not in ("a", "an", "the", "of", "to", "in", "on", "and", "her", "his", "their")]
    a, g = tk(a or ""), tk(g)
    c = sum((collections.Counter(a) & collections.Counter(g)).values())
    return 0.0 if not c else 2 * c / (len(a) + len(g))

rows = []
for sample in locomo.split_samples(json.load(open(sys.argv[1])), "dev"):
    user, m2d = locomo.ingest(mm, sample)
    for qa in sample["qa"]:
        if qa["category"] == 5:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            out = mm.retrieve(user, qa["question"], limit=10)
        a = out.get("answer") or {}
        if a.get("later_answer"):
            rows.append({"q": qa["question"], "cat": qa["category"], "gold": str(qa.get("answer")),
                         "confident": a["text"], "later": a["later_answer"],
                         "f1_confident": round(f1(a["text"], qa.get("answer")), 3), "f1_later": round(f1(a["later_answer"], qa.get("answer")), 3)})
n = len(rows)
summ = {"conflicts": n,
        "confident_better": sum(r["f1_confident"] > r["f1_later"] for r in rows),
        "later_better": sum(r["f1_later"] > r["f1_confident"] for r in rows),
        "tie": sum(r["f1_later"] == r["f1_confident"] for r in rows),
        "mean_f1_confident": round(sum(r["f1_confident"] for r in rows) / max(n, 1), 3),
        "mean_f1_later": round(sum(r["f1_later"] for r in rows) / max(n, 1), 3),
        "by_cat": {c: sum(1 for r in rows if r["cat"] == c) for c in (1, 2, 3, 4)}}
json.dump({"summary": summ, "rows": rows}, open(sys.argv[2], "w"), indent=1)
print(json.dumps(summ))
