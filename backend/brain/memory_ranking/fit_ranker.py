"""
Fit the recall relevance model and its abstention threshold.

Training data: groups (data sources) of queries; each query is a list of
(features, label) for its candidates (label 1 = the memory answers the query).
A query with no positive candidate is unanswerable.

  1. weights: MAP logistic regression around the documented prior
     (ranker_model.prior.json). group_weighting="by_size" (default) weights every
     query equally, so larger sources count more; "equal" gives every group the
     same total weight. by_size is the default because the larger, independent
     sources are the more reliable signal than a few dozen hand-written queries.
  2. regularisation strength: chosen by leave-one-group-out cross-validation on
     ranking quality (group-averaged hit@k on answerable queries, k = ANSWER_K).
  3. abstention threshold: an explicit policy instead of a fitted trade-off.
     Recall abstains only when the top relevance is below the value under which
     at most `max_false_abstain_rate` of answerable queries fall, measured on
     out-of-fold scores (config.yaml recall.max_false_abstain_rate).
     The resulting decline rate on unanswerable queries is reported, not optimised.

Sources:
  --from-db   feedback logged via POST /memory/feedback (RecallLog)
  eval        `python eval/run_eval.py --fit-ranker [--locomo PATH]` fits on the
              eval dev split (+ LoCoMo dev conversations) and calls fit_model_cv().

Writes <storage>/ranker_model.json (picked up without restart).
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np

from backend import settings
from backend.brain.memory_ranking import logistic
from backend.brain.memory_ranking.ranker import FEATURES, Ranker, RecallLog, model_path, PRIOR_MODEL_PATH

ANSWER_K = 5  # results served by default; ranking quality is judged on these


def prior_weights():
    with open(PRIOR_MODEL_PATH) as f:
        return json.load(f)["weights"]


def max_false_abstain_rate():
    return float(settings.get("recall.max_false_abstain_rate", 0.05))


def _top_and_hit(cands, weights, k=ANSWER_K):
    """(top relevance, evidence within top-k, answerable) for one query."""
    model = {"weights": weights, "threshold": 0.0}
    scored = sorted(((Ranker.score(f, model), y) for f, y in cands), key=lambda t: -t[0])
    return scored[0][0], any(y for _, y in scored[:k]), any(y for _, y in scored)


def _balanced(groups, names, equal_groups=True):
    """Candidate features/labels with weights; equal_groups gives every group the same total weight."""
    feats, labels, cw = [], [], []
    for g in names:
        n_cands = sum(len(q) for q in groups[g]) or 1
        for q in groups[g]:
            for f, y in q:
                feats.append(f)
                labels.append(y)
                cw.append(1.0 / n_cands if equal_groups else 1.0)
    return feats, labels, cw


def threshold_for(outcomes_by_group, rate):
    """
    Largest threshold t such that, averaged over groups, at most `rate` of answerable
    queries have top relevance below t (i.e. would be wrongly declined).
    """
    answerable = {g: np.sort([p for p, _, ans in os_ if ans]) for g, os_ in outcomes_by_group.items()}
    answerable = {g: a for g, a in answerable.items() if len(a)}
    if not answerable:
        return 0.0
    candidates = np.unique(np.concatenate(list(answerable.values())))

    def false_rate(t):
        return float(np.mean([np.mean(a < t) for a in answerable.values()]))
    ok = [t for t in candidates if false_rate(t) <= rate]
    return float(max(ok)) if ok else 0.0


def decline_rates(outcomes_by_group, t):
    """Group-averaged share of answerable / unanswerable queries declined at threshold t."""
    ans = [np.mean([p < t for p, _, a in os_ if a]) for os_ in outcomes_by_group.values() if any(a for *_, a in os_)]
    una = [np.mean([p < t for p, _, a in os_ if not a]) for os_ in outcomes_by_group.values()
           if any(not a for *_, a in os_)]
    return (float(np.mean(ans)) if ans else float("nan"), float(np.mean(una)) if una else float("nan"))


def fit_model_cv(groups, lams=None, source="", max_candidates=None, group_weighting="by_size"):
    """groups: {group_name: [query candidates]}. Returns the model dict (see module docstring)."""
    if group_weighting not in ("by_size", "equal"):
        raise ValueError("group_weighting must be 'by_size' or 'equal'")
    equal = group_weighting == "equal"
    prior = prior_weights()
    if max_candidates:
        model = {"weights": prior, "threshold": 0.0}
        groups = {g: [sorted(q, key=lambda fy: -Ranker.score(fy[0], model))[:max_candidates] for q in qs]
                  for g, qs in groups.items()}
    names = list(groups)
    if lams is None:
        # The prior's pull is lam / n, so the grid scales with the number of training candidates.
        n = sum(len(q) for g in names for q in groups[g])
        lams = tuple(float(f"{n * r:.4g}") for r in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0))  # never rounds to 0

    best, cv_table = None, {}
    for lam in lams:
        oof = {}
        for held in names:
            f, y, w = _balanced(groups, [g for g in names if g != held], equal)
            weights = logistic.fit(f, y, prior, FEATURES, lam=lam, sample_weight=w)
            oof[held] = [_top_and_hit(q, weights) for q in groups[held] if q]  # never seen by this fit
        if not equal:  # pool all queries: every query counts the same
            oof = {"all": [o for os_ in oof.values() for o in os_]}
        hit = float(np.mean([np.mean([h for _, h, a in os_ if a]) for os_ in oof.values() if any(a for *_, a in os_)]))
        cv_table[lam] = round(hit, 4)
        if best is None or round(hit, 4) >= round(best[1], 4):  # lams ascend: ties prefer the stronger prior
            best = (lam, hit, oof)
    lam, cv_hit, oof = best

    rate = max_false_abstain_rate()
    threshold = threshold_for(oof, rate)
    false_decline, true_decline = decline_rates(oof, threshold)
    f, y, w = _balanced(groups, names, equal)
    weights = logistic.fit(f, y, prior, FEATURES, lam=lam, sample_weight=w)
    return {"weights": weights, "threshold": threshold, "lam": lam, "group_weighting": group_weighting,
            "cv_hit_at_k": cv_hit, "cv_by_lam": cv_table, "answer_k": ANSWER_K,
            "max_false_abstain_rate": rate,
            "cv_answerable_declined": false_decline, "cv_unanswerable_declined": true_decline,
            "n_queries": sum(len(groups[g]) for g in names), "n_candidates": len(y),
            "groups": names, "max_candidates_per_query": max_candidates,
            "selection": "leave-one-group-out CV; lam by hit@k; "
                         "threshold = max_false_abstain_rate quantile of out-of-fold answerable top scores",
            "source": source, "fitted_at": time.time()}


def fit_model(queries, lam=1.0, source=""):
    """Single-group fit (e.g. logged feedback): weights + the same threshold policy, in-sample."""
    prior = prior_weights()
    f = [x for q in queries for x, _ in q]
    y = [lbl for q in queries for _, lbl in q]
    weights = logistic.fit(f, y, prior, FEATURES, lam=lam)
    outcomes = {"all": [_top_and_hit(q, weights) for q in queries if q]}
    threshold = threshold_for(outcomes, max_false_abstain_rate())
    false_decline, true_decline = decline_rates(outcomes, threshold)
    return {"weights": weights, "threshold": threshold, "lam": lam,
            "train_answerable_declined": false_decline, "train_unanswerable_declined": true_decline,
            "max_false_abstain_rate": max_false_abstain_rate(),
            "n_queries": len(queries), "n_candidates": len(y), "source": source, "fitted_at": time.time()}


def save(model):
    with open(model_path(), "w") as f:
        json.dump(model, f, indent=2)
    return model_path()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-db", action="store_true", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--min-queries", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = defaultdict(list)
    for recall_id, features, label in RecallLog().labelled():
        groups[recall_id].append((features, label))
    queries = list(groups.values())
    print(f"{len(queries)} labelled recalls")
    if len(queries) < args.min_queries:
        raise SystemExit(f"Need >= {args.min_queries} recalls with feedback to fit.")
    model = fit_model(queries, lam=args.lam, source="feedback")
    print(json.dumps({k: v for k, v in model.items() if k != "weights"}, indent=2))
    for k in ["bias"] + FEATURES:
        print(f"  {k:18} {Ranker.model()['weights'][k]:+.2f} -> {model['weights'][k]:+.2f}")
    if not args.dry_run:
        print("wrote", save(model))


if __name__ == "__main__":
    main()
