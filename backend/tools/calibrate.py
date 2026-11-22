"""
Calibrate the answer check for THIS deployment, with a guarantee on wrong refusals.

    python -m backend.tools.calibrate --from-db                  # from feedback your users gave
    python -m backend.tools.calibrate --from-file sample.json    # from a labelled sample
          [--alpha 0.05] [--delta 0.05] [--dry-run]

The result (a certificate) is stored under the storage root and used by every recall
from then on; recall receipts quote it. See backend/brain/features/calibration.py for
the statistics. In short: with probability >= 1 - delta, at most alpha of answerable
questions *like the calibration questions* are wrongly declined.

--from-db uses recalls where feedback marked one of the checked memories relevant.
Caveat: only questions your users saw answers for can receive such feedback, so a
deployment whose threshold already declines too often under-collects exactly the
cases it gets wrong. Prefer --from-file for the first calibration.

--from-file: JSON list of scenarios, the same shape as eval/data/*/retrieval.jsonl
items: {"memories": ["..."], "queries": [{"q": "...", "expect": "text in the answer memory"}]}.
Queries with "expect": null (unanswerable) are skipped: the guarantee concerns
answerable questions. Runs in a sandbox; production data is untouched.
"""
import argparse
import json
import sys


def margins_from_file(path, embedder):
    from backend import settings
    from backend.tools.sandbox import Sandbox
    scenarios = json.load(open(path))
    margins, skipped = [], 0
    with Sandbox(embedder=embedder, answer_check=True) as box:
        from backend.brain.features.answer_check import AnswerChecker
        cfg = settings.config().setdefault("recall", {}).setdefault("answer_check", {})
        cfg["threshold"] = -1e9  # measure margins; never gate while calibrating
        checker = AnswerChecker.from_config()
        model = checker.model_name if checker else None
        for n, sc in enumerate(scenarios):
            user = f"calib_{n}"
            with box.silenced():
                for i, m in enumerate(sc["memories"]):
                    box.mm.store(user, m, timestamp=1_700_000_000 + i)
            for q in sc["queries"]:
                if not q.get("expect"):
                    continue
                with box.silenced():
                    out = box.mm.retrieve(user, q["q"], limit=5)
                served = [m["content"] for m in out["memories"]]
                chk = out["debug"].get("answer_check")
                if chk and any(q["expect"].lower() in c.lower() for c in served):
                    margins.append(chk["best_margin"])
                else:
                    skipped += 1  # evidence not retrieved: a ranking miss, not a refusal decision
    return margins, model, skipped


def margins_from_db():
    from backend.brain.features.answer_check import AnswerChecker
    from backend.brain.memory_ranking.ranker import RecallLog
    checker = AnswerChecker.from_config()
    if checker is None:
        raise SystemExit("the answer check is disabled or unavailable (config.yaml recall.answer_check)")
    return RecallLog().answerable_margins(checker.model_name), checker.model_name, 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", action="store_true")
    src.add_argument("--from-file")
    ap.add_argument("--alpha", type=float, default=0.05, help="max wrong-refusal rate")
    ap.add_argument("--delta", type=float, default=0.05, help="allowed probability that the bound fails")
    ap.add_argument("--embedder", default="configured", help="for --from-file: configured | hash | st:<model>")
    ap.add_argument("--dry-run", action="store_true", help="print the certificate without saving it")
    args = ap.parse_args(argv)

    from backend.brain.features import calibration
    margins, model, skipped = margins_from_file(args.from_file, args.embedder) if args.from_file else margins_from_db()
    if not model:
        raise SystemExit("no answer-check model in use; nothing to calibrate")
    cert = calibration.certify_threshold(margins, args.alpha, args.delta)
    print(json.dumps({**cert, "model": model, "skipped_not_retrieved": skipped}, indent=2))
    if not args.dry_run:
        print("saved", calibration.save(cert, model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
