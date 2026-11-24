"""Development measurement for the retrieval fixes (neighbour context, declines, second search step).

    python measure.py LOCOMO10.json OUT.json --convs conv-44          # or --split dev

Per question: evidence recall@10 of what is served and of the ranking before the answer check,
the rank of the first evidence message among all candidates, and whether recall declined. No LLM.
Config variants are selected with RECALLIX_CONFIG.
"""
import argparse, contextlib, io, json, sys
sys.path[:0] = [".", "eval", "eval/benchmarks"]
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--convs", default=""); ap.add_argument("--split", default="")
args = ap.parse_args()
import run_eval, locomo
env = run_eval.Env("st:BAAI/bge-small-en-v1.5")
from backend.api.deps import get_memory_manager, reset
reset()
with contextlib.redirect_stdout(io.StringIO()):
    mm = get_memory_manager()
data = json.load(open(args.data))
samples = locomo.split_samples(data, args.split) if args.split else [s for s in data if s["sample_id"] in args.convs.split(",")]
rows = []
for sample in samples:
    user, m2d = locomo.ingest(mm, sample)
    known = set(d for d, _, _ in locomo.turns_of(sample))
    for qi, qa in enumerate(sample["qa"]):
        with contextlib.redirect_stdout(io.StringIO()):
            out = mm.retrieve(user, qa["question"], limit=10, return_candidates=True)
        ev = locomo.norm_evidence(qa.get("evidence")) & known
        served = [d for m in out["memories"] for d in m2d[m["id"]]]
        cands = sorted(out["_candidates"], key=lambda c: -c["relevance"])
        ranked = [d for c in cands for d in m2d[c["id"]]]
        first = next((i + 1 for i, c in enumerate(cands) if set(m2d[c["id"]]) & ev), None)
        chk = out["debug"].get("answer_check") or {}
        rows.append({"conv": sample["sample_id"], "qi": qi, "cat": qa["category"], "status": out["status"],
                     "declined": out["status"] in ("ABSTAINED", "NO_MATCH"), "n_ev": len(ev),
                     "served10": (len(set(served) & ev) / len(ev)) if ev else None,
                     "ranked10": (len(set(ranked[:10]) & ev) / len(ev)) if ev else None,
                     "first_ev_rank": first, "n_cands": len(cands), "margin": chk.get("best_margin"),
                     "threshold": chk.get("threshold"), "outsider": bool(out["receipt"].get("unknown_person")) if isinstance(out.get("receipt"), dict) else None,
                     "question": qa["question"], "latency_ms": out["latency_ms"]})
ans = [r for r in rows if r["cat"] != 5 and r["n_ev"]]; adv = [r for r in rows if r["cat"] == 5]
m = lambda xs: round(float(np.mean(xs)), 4) if len(xs) else None
names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
res = {"questions": len(rows), "served_recall@10": m([r["served10"] for r in ans]), "ranked_recall@10": m([r["ranked10"] for r in ans]),
       "served_any_evidence": f'{sum(r["served10"] > 0 for r in ans)}/{len(ans)}',
       "served_recall@10_by_cat": {names[c]: m([r["served10"] for r in ans if r["cat"] == c]) for c in names},
       "answerable_wrongly_declined": f'{sum(r["declined"] for r in ans)}/{len(ans)}',
       "traps_declined": f'{sum(r["declined"] for r in adv)}/{len(adv)}',
       "recall_latency_ms_median": m([float(np.median([r["latency_ms"] for r in rows]))])}
json.dump({"summary": res, "rows": rows}, open(args.out, "w"), indent=1)
print(json.dumps(res))
