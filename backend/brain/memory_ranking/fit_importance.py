"""
Fit importance weights (MAP logistic regression around the current defaults).

Label sources:
  --from-db     real usage. A stored memory older than --min-age-days is a
                positive if recall ever returned it (retrieval_count > 0),
                negative otherwise. Uses the feature vector saved in the
                memory's metadata at store time.
  --from-file   JSONL of {"features": {...}, "keep": 0/1}.

Writes <storage>/importance_weights.json, which ImportanceRanker picks up
automatically (no restart needed).

  python -m backend.brain.memory_ranking.fit_importance --from-db --dry-run
"""
import argparse
import json
import sqlite3
import time

from backend.brain.memory_ranking import logistic
from backend.brain.memory_ranking.importance_ranker import WEIGHTS, FEATURES, weights_path


def fit_weights(feature_dicts, labels, prior=None, lam=0.5):
    from backend.brain.memory_ranking.importance_ranker import PRIOR_WEIGHTS_PATH
    if prior is None:
        with open(PRIOR_WEIGHTS_PATH) as f:
            prior = json.load(f)["weights"]
    return logistic.fit(feature_dicts, labels, prior, FEATURES, lam=lam)


def rows_from_db(db_path, min_age_days):
    cutoff = time.time() - min_age_days * 86400
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT metadata, retrieval_count FROM episodic_events WHERE timestamp < ?", (cutoff,)).fetchall()
    conn.close()
    feats, labels = [], []
    for metadata, count in rows:
        try:
            f = json.loads(metadata or "{}").get("importance_features")
        except ValueError:
            f = None
        if f and all(k in f for k in FEATURES):
            feats.append(f)
            labels.append(1 if (count or 0) > 0 else 0)
    return feats, labels


def main():
    from backend.utils.paths import get_db_path, ensure_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", action="store_true")
    src.add_argument("--from-file")
    ap.add_argument("--db", default=None)
    ap.add_argument("--min-age-days", type=float, default=7.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--min-examples", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.from_db:
        feats, labels = rows_from_db(args.db or get_db_path("backend/storage/sqlite_db/memory.db"), args.min_age_days)
    else:
        with open(args.from_file) as f:
            items = [json.loads(line) for line in f if line.strip()]
        feats, labels = [i["features"] for i in items], [i["keep"] for i in items]

    pos = sum(labels)
    print(f"{len(labels)} examples ({pos} positive, {len(labels) - pos} negative)")
    if len(labels) < args.min_examples or pos == 0 or pos == len(labels):
        raise SystemExit(f"Not enough labelled data to fit (need >= {args.min_examples} with both classes).")

    fitted = fit_weights(feats, labels, lam=args.lam)
    print(f"log-loss  current={logistic.log_loss(feats, labels, WEIGHTS, FEATURES):.3f}  "
          f"fitted={logistic.log_loss(feats, labels, fitted, FEATURES):.3f}")
    for k in ["bias"] + FEATURES:
        print(f"  {k:12} {WEIGHTS[k]:+.2f} -> {fitted[k]:+.2f}")
    if args.dry_run:
        return
    path = weights_path()
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump({"weights": fitted, "n_examples": len(labels), "fitted_at": time.time(),
                   "source": "db" if args.from_db else args.from_file, "lam": args.lam}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
