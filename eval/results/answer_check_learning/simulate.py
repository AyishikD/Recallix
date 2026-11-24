"""
Simulated deployments for the learned answer-check threshold (backend/brain/features/answer_check_learning.py).

Per-question rows from a recall run (longmemeval.py --rows; retrieval_fixes/measure.py), with the margin the
threshold acts on (counted_margin). For each of 20 random halvings: the feedback a deployment would collect on
half A (answerable questions whose evidence was returned -> "used"; trap/unanswerable -> "none_used") goes
through the real code (choose, score, the held-out adoption rule); the result is measured on half B, which it
never saw. A question declined by a rule rather than by the threshold keeps its decision.

    python eval/results/answer_check_learning/simulate.py      # writes results.json here
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from backend.brain.features.answer_check_learning import choose, score  # noqa: E402
from backend.brain.memory_ranking.self_improve import held_out  # noqa: E402

COST, SPLITS, HOLDOUT = 3.0, 20, 0.2


def load(path):
    rows = json.load(open(path))["rows"]
    t0 = rows[0]["threshold"]
    out = []
    for x in rows:
        m = x["counted_margin"] if x.get("counted_margin") is not None else x["margin"]
        fixed = None
        if x["declined"] and (m is None or m >= t0):
            fixed = True       # declined by a rule, not the threshold
        if not x["declined"] and m is not None and m < t0:
            fixed = False      # answered by a rule (calculation question)
        trap = x["unanswerable"] if "qid" in x else x["cat"] == 5
        seen = x["evidence_returned"] if "qid" in x else (x.get("first_ev_rank") is not None and x["first_ev_rank"] <= 10)
        out.append({"trap": trap, "m": m, "fixed": fixed, "seen": seen})
    return out, t0


def declined(r, t):
    return r["fixed"] if r["fixed"] is not None else (r["m"] is None or r["m"] < t)


def simulate(rows, t0):
    agg, thresholds, adopted = {k: [] for k in ("ans_before", "ans_after", "trap_before", "trap_after",
                                                "score_before", "score_after")}, [], 0
    for s in range(SPLITS):
        rnd = random.Random(s)
        idx = list(range(len(rows)))
        rnd.shuffle(idx)
        A, B = [rows[i] for i in idx[: len(rows) // 2]], [rows[i] for i in idx[len(rows) // 2:]]
        fb = [(f"{s}-{i}", r["m"], not r["trap"]) for i, r in enumerate(A) if r["m"] is not None and (r["trap"] or r["seen"])]
        test = [x for x in fb if held_out(x[0], HOLDOUT)]
        train = [x for x in fb if not held_out(x[0], HOLDOUT)]
        cand = choose(train, t0, COST)
        ok = score(test, cand, COST) > score(test, t0, COST)
        t = choose(fb, t0, COST) if ok else t0
        adopted += ok
        thresholds.append(t)
        ans = [r for r in B if not r["trap"] and r["seen"]]
        traps = [r for r in B if r["trap"]]
        for tag, tt in (("before", t0), ("after", t)):
            agg[f"ans_{tag}"].append(st.mean(not declined(r, tt) for r in ans))
            agg[f"trap_{tag}"].append(st.mean(not declined(r, tt) for r in traps))
            agg[f"score_{tag}"].append(sum(not declined(r, tt) for r in ans) - COST * sum(not declined(r, tt) for r in traps))
    return {"shipped_threshold": t0, "adopted_in_splits": f"{adopted}/{SPLITS}", "learned_threshold_median": st.median(thresholds),
            "answerable_with_evidence_answered": {"before": round(st.mean(agg["ans_before"]), 3), "after": round(st.mean(agg["ans_after"]), 3)},
            "trap_or_unanswerable_answered": {"before": round(st.mean(agg["trap_before"]), 3), "after": round(st.mean(agg["trap_after"]), 3)},
            "score_per_half": {"before": round(st.mean(agg["score_before"]), 1), "after": round(st.mean(agg["score_after"]), 1)}}


if __name__ == "__main__":
    out = {"wrong_answer_cost": COST, "splits": SPLITS}
    for name, f in (("longmemeval", "longmemeval_rows.json"), ("locomo_dev", "locomo_dev_rows.json")):
        out[name] = simulate(*load(HERE / f))
    json.dump(out, open(HERE / "results.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
