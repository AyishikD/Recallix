"""
Fit the recall relevance model and its abstention threshold.

Training data is a list of queries, each a list of (features, label) for its
candidates (label 1 = the memory answers the query). A query whose candidates
are all 0 is unanswerable: the right behaviour is to abstain.

  1. weights: MAP logistic regression over all candidates (prior = current model)
  2. threshold: the value t that maximises per-query correctness, where
       answerable query   correct iff top candidate is relevant and p_top >= t
       unanswerable query correct iff p_top < t
     chosen as the midpoint between adjacent sorted p_top values.

Sources:
  --from-db   feedback logged via POST /memory/feedback (RecallLog)
  eval        `python eval/run_eval.py --fit-ranker` builds queries from the
              eval dev split and calls fit_model().

Writes <storage>/ranker_model.json (picked up without restart).
"""
import argparse
import json
import time
from collections import defaultdict

from backend.brain.memory_ranking import logistic
from backend.brain.memory_ranking.ranker import FEATURES, Ranker, RecallLog, model_path


def choose_threshold(queries, weights):
    """queries: list of [(features, label)]. Returns (threshold, accuracy)."""
    model = {"weights": weights, "threshold": 0.0}
    tops = []
    for cands in queries:
        if not cands:
            continue
        scored = [(Ranker.score(f, model), y) for f, y in cands]
        p, y = max(scored, key=lambda t: t[0])
        tops.append((p, bool(y), any(lbl for _, lbl in cands)))
    if not tops:
        return 0.5, float("nan")
    ps = sorted({p for p, _, _ in tops})
    options = [ps[0] - 1e-6] + [(a + b) / 2 for a, b in zip(ps, ps[1:])] + [ps[-1] + 1e-6]

    def acc(t):
        return sum((ok and p >= t) if answerable else (p < t) for p, ok, answerable in tops) / len(tops)
    best = max(options, key=lambda t: (acc(t), -abs(t - 0.5)))
    return float(best), acc(best)


def prior_weights():
    import os
    with open(os.path.join(os.path.dirname(__file__), "ranker_model.prior.json")) as f:
        return json.load(f)["weights"]


def fit_model_cv(groups, lams=(0.3, 1.0, 3.0, 10.0, 30.0), source=""):
    """
    groups: {group_name: [query candidates]} (e.g. one group per eval scenario).
    Chooses the regularisation strength and threshold by leave-one-group-out
    cross-validation (out-of-fold predictions), then fits the final weights on
    all groups with that strength. Ties prefer stronger regularisation.
    """
    prior = prior_weights()
    names = list(groups)
    best = None
    cv_table = {}
    for lam in lams:
        oof = []  # (features, label) lists scored by a model that never saw them
        oof_models = []
        for held in names:
            train = [q for g in names if g != held for q in groups[g]]
            feats = [f for q in train for f, _ in q]
            labels = [y for q in train for _, y in q]
            w = logistic.fit(feats, labels, prior, FEATURES, lam=lam)
            oof_models.append((held, w))
        # threshold from out-of-fold top scores
        tops = []
        for held, w in oof_models:
            model = {"weights": w, "threshold": 0.0}
            for cands in groups[held]:
                if not cands:
                    continue
                p, y = max(((Ranker.score(f, model), lbl) for f, lbl in cands), key=lambda t: t[0])
                tops.append((p, bool(y), any(lbl for _, lbl in cands)))
        ps = sorted({p for p, _, _ in tops})
        options = [ps[0] - 1e-6] + [(a + b) / 2 for a, b in zip(ps, ps[1:])] + [ps[-1] + 1e-6]
        acc = lambda t: sum((ok and p >= t) if ans else (p < t) for p, ok, ans in tops) / len(tops)  # noqa: E731
        t_best = max(options, key=lambda t: (acc(t), -abs(t - 0.5)))
        cv_table[lam] = round(acc(t_best), 4)
        if best is None or acc(t_best) >= best[1]:
            best = (lam, acc(t_best), t_best)
    lam, cv_acc, threshold = best
    all_q = [q for g in names for q in groups[g]]
    weights = logistic.fit([f for q in all_q for f, _ in q], [y for q in all_q for _, y in q], prior, FEATURES, lam=lam)
    return {"weights": weights, "threshold": float(threshold), "cv_query_accuracy": cv_acc, "cv_by_lam": cv_table,
            "lam": lam, "n_queries": len(all_q), "n_candidates": sum(len(q) for q in all_q),
            "selection": "leave-one-group-out CV (lam and threshold from out-of-fold predictions)",
            "source": source, "fitted_at": time.time()}


def fit_model(queries, prior=None, lam=1.0, source=""):
    prior = prior or prior_weights()
    feats = [f for q in queries for f, _ in q]
    labels = [y for q in queries for _, y in q]
    weights = logistic.fit(feats, labels, prior, FEATURES, lam=lam)
    threshold, accuracy = choose_threshold(queries, weights)
    return {"weights": weights, "threshold": threshold, "train_query_accuracy": accuracy,
            "n_queries": len(queries), "n_candidates": len(labels), "lam": lam,
            "source": source, "fitted_at": time.time()}


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
