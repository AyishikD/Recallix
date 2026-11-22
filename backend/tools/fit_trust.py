"""
Fit and validate the trust score on LoCoMo dev conversations (never the test split).

    python -m backend.tools.fit_trust --locomo locomo10.json [--embedder st:...] [--dry-run]

For every dev question, the top served memory is labelled correct if it holds an evidence
turn of an answerable question; for adversarial questions any served memory is wrong.
Calibration is measured with leave-one-conversation-out cross-validation (each conversation
is scored by a model fitted on the others): Brier score, expected calibration error (10 bins)
and AUC, next to plain relevance as the baseline. The model is written to
backend/brain/features/trust.default.json only if it is better calibrated than relevance.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def collect(locomo_path, embedder):
    sys.path.insert(0, str(REPO / "eval" / "benchmarks"))
    import locomo
    from backend.tools.sandbox import Sandbox
    rows = []  # (conversation, features, label, relevance)
    with Sandbox(embedder=embedder, answer_check=True) as box:
        from backend.brain.features.trust import features
        for sample in locomo.split_samples(json.load(open(locomo_path)), "dev"):
            user, mem_to_dia = locomo.ingest(box.mm, sample)
            for qa in sample["qa"]:
                evidence = locomo.norm_evidence(qa.get("evidence")) if qa["category"] != 5 else set()
                with box.silenced():
                    out = box.mm.retrieve(user, qa["question"], limit=5)
                if not out["memories"]:
                    continue  # nothing served: nothing to trust
                top = out["memories"][0]
                label = int(bool(evidence) and bool(set(mem_to_dia.get(top["id"], [])) & evidence))
                rows.append((sample["sample_id"], features(top), label, float(top.get("relevance") or 0.0)))
    return rows


def calibration(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())) if pos.any() and (~pos).any() else float("nan")
    return {"brier": round(brier, 4), "ece": round(float(ece), 4), "auc": round(auc, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locomo", required=True)
    ap.add_argument("--embedder", default="st:sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from backend.brain.features import trust
    from backend.brain.memory_ranking import logistic
    rows = collect(args.locomo, args.embedder)
    convs = sorted({c for c, *_ in rows})
    prior = {k: 0.0 for k in ["bias"] + trust.FEATURES}
    oof = np.zeros(len(rows))
    for c in convs:  # leave one conversation out
        tr = [i for i, r in enumerate(rows) if r[0] != c]
        te = [i for i, r in enumerate(rows) if r[0] == c]
        w = logistic.fit([rows[i][1] for i in tr], [rows[i][2] for i in tr], prior, trust.FEATURES, lam=1.0)
        oof[te] = logistic.predict([rows[i][1] for i in te], w, trust.FEATURES)
    y = [r[2] for r in rows]
    report = {"questions_with_a_served_memory": len(rows), "correct_rate": round(float(np.mean(y)), 4),
              "trust_cross_validated": calibration(oof, y),
              "relevance_as_probability": calibration([r[3] for r in rows], y)}
    weights = logistic.fit([r[1] for r in rows], y, prior, trust.FEATURES, lam=1.0)
    print(json.dumps(report, indent=1))
    better = report["trust_cross_validated"]["brier"] < report["relevance_as_probability"]["brier"]
    if args.dry_run or not better:
        print("not saved" + ("" if better else ": not better calibrated than relevance"))
        return 0
    Path(trust.DEFAULT_PATH).write_text(json.dumps({
        "weights": weights, "features": trust.FEATURES, "validation": report,
        "fitted_on": "LoCoMo dev conversations (0-4), leave-one-conversation-out", "embedder": args.embedder}, indent=1))
    print("saved", trust.DEFAULT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
