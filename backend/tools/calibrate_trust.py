"""
Recalibrate the trust score for THIS deployment, from answers your users confirmed.

    python -m backend.tools.calibrate_trust --from-db                 # recalls that got feedback
    python -m backend.tools.calibrate_trust --from-file sample.json   # a labelled sample, in a sandbox
          [--min-samples 50] [--dry-run]

Why: the trust model is fitted on LoCoMo. On new LoCoMo conversations it is well calibrated, but
on a different kind of conversation (LongMemEval) its probabilities were about half the true rate
(eval/results/trust_check_*.json). Its ranking of answers carries over better than its level, so
only the level is refitted here: trust = sigmoid(a*z + b) on the model's log-odds z (two numbers,
pulled towards "no change" when there are few labels).

The fit is checked by 5-fold cross-validation before it is saved: it is kept only if it lowers
the Brier score on held-out labels. Recall receipts then say the score is calibrated for this
deployment (receipt.trust_calibration) instead of carrying the reference calibration.

--from-db: every recall logs its top memory's log-odds; feedback (POST /memory/feedback or the
MCP confirm_answer tool) labels it 1 if that memory was confirmed as the answer, else 0.
Caveat: users mostly give feedback on answers they saw, which can over-represent good answers.

--from-file: the same scenario format as backend.tools.calibrate:
[{"memories": ["..."], "queries": [{"q": "...", "expect": "text in the answer memory" | null}]}]
The top served memory is labelled 1 if it contains `expect`; for expect null any served memory is 0.
"""
import argparse
import json
import sys
import time


def samples_from_file(path, embedder):
    from backend.tools.sandbox import Sandbox
    scenarios = json.load(open(path))
    out = []
    with Sandbox(embedder=embedder, answer_check=True) as box:
        from backend.brain.features import trust
        for n, sc in enumerate(scenarios):
            user = f"trustcal_{n}"
            with box.silenced():
                for i, m in enumerate(sc["memories"]):
                    box.mm.store(user, m, timestamp=1_700_000_000 + i)
            for q in sc["queries"]:
                with box.silenced():
                    res = box.mm.retrieve(user, q["q"], limit=5)
                if not res["memories"]:
                    continue
                top = res["memories"][0]
                z = trust.logit(top)
                if z is not None:
                    expect = q.get("expect")
                    out.append((z, int(bool(expect) and expect.lower() in top["content"].lower())))
    return out


def samples_from_db():
    from backend.brain.memory_ranking.ranker import RecallLog
    return RecallLog().trust_samples()


def _metrics(p, y):
    from backend.tools.fit_trust import calibration
    return calibration(p, y)


def cross_validate(samples, folds=5):
    """Held-out predictions before and after recalibration (each fold fitted on the others)."""
    from backend.brain.features import trust
    before, after, ys = [], [], []
    for k in range(folds):
        train = [s for i, s in enumerate(samples) if i % folds != k]
        test = [s for i, s in enumerate(samples) if i % folds == k]
        a, b = trust.fit_recalibration([z for z, _ in train], [y for _, y in train])
        for z, y in test:
            before.append(trust.recalibrated(z, 1.0, 0.0))
            after.append(trust.recalibrated(z, a, b))
            ys.append(y)
    return _metrics(before, ys), _metrics(after, ys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", action="store_true")
    src.add_argument("--from-file")
    ap.add_argument("--embedder", default="st:sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from backend.brain.features import trust
    if not trust.model():
        print("no trust model installed; nothing to calibrate")
        return 1
    samples = samples_from_db() if args.from_db else samples_from_file(args.from_file, args.embedder)
    pos = sum(y for _, y in samples)
    print(f"{len(samples)} labelled answers ({pos} right, {len(samples) - pos} wrong)")
    if len(samples) < args.min_samples or min(pos, len(samples) - pos) < 10:
        print(f"not enough labels: need >= {args.min_samples}, with at least 10 right and 10 wrong")
        return 1
    before, after = cross_validate(samples)
    a, b = trust.fit_recalibration([z for z, _ in samples], [y for _, y in samples])
    report = {"n": len(samples), "right": pos, "source": "feedback" if args.from_db else args.from_file,
              "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "cross_validated": {"reference_calibration": before, "recalibrated": after}}
    print(json.dumps({"a": round(a, 4), "b": round(b, 4), **report}, indent=1))
    if after["brier"] >= before["brier"]:
        print("not saved: recalibration does not improve held-out Brier score")
        return 0
    if args.dry_run:
        print("dry run: not saved")
        return 0
    print("saved", trust.save_calibration(a, b, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
