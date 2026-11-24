"""
Learning the answer check's decline threshold from use (config.yaml `learning.answer_check`, off by default).

Why: the answer check declines when its best margin is below a threshold. No single threshold suits every
deployment. Where nearly every question has an answer in memory, a high threshold wrongly declines many of them;
where many questions are traps (asked about things never said), a low one answers them. Measured on saved runs
(simulating a deployment that calibrates on half its questions and is judged on the other half, a wrong answer
weighted as 3 refusals): LongMemEval (6% unanswerable) wrongly declined 17% -> 3% of answerable questions;
LoCoMo (about 24% traps) moved the other way, declining more traps (33% -> 47%).

Data: recalls with feedback (POST /memory/feedback, SDK `used` / `none_used`), with the answer check's margin:
  answerable      a memory the answer check read was marked used
  not answerable  nothing was used ("none_used")
A declined recall returns nothing unless the caller asked for keep_on_decline, so wrong refusals can only be
seen (and the threshold only learn to go down) when callers use keep_on_decline and give feedback on declines.

Each run (with the ranker's, workers.learning_hours):
  1. needs `min_answerable` + `min_unanswerable` labelled recalls, and `min_new` new ones since the last attempt.
  2. sets aside the `learning.holdout` share (same split as the ranker: hash of the recall id) and picks, on
     the rest, the threshold with the best score: +1 per answerable recall answered, -wrong_answer_cost per
     not-answerable recall answered (a wrong answer is worse than a refusal). Ties keep the threshold nearest
     the one in use.
  3. adopts it only if it also scores better than the threshold in use on the held-out recalls; the threshold
     is then refitted on all of them. The previous one is kept for rollback.

Resolution order in AnswerChecker.threshold(): config override > certificate (backend/tools/calibrate.py) >
this learned threshold > the dev-fitted default. Every run is recorded in the learning log
(GET /internal/learning). Only this deployment's own feedback is used; nothing leaves the server.

    python -m backend.brain.features.answer_check_learning [--dry-run | --rollback]
"""
import argparse
import json
import os
import time

from backend import settings

LEARNED_FILE = "answer_check_learned.json"


def config():
    c = (settings.get("learning", {}) or {}).get("answer_check", {}) or {}
    return {"enabled": bool(c.get("enabled", False)),
            "wrong_answer_cost": float(c.get("wrong_answer_cost", 3.0)),
            "min_answerable": int(c.get("min_answerable", 50)),
            "min_unanswerable": int(c.get("min_unanswerable", 20)),
            "min_new": int(c.get("min_new", 20)),
            "min_holdout_answerable": int(c.get("min_holdout_answerable", 6)),
            "min_holdout_unanswerable": int(c.get("min_holdout_unanswerable", 3))}


def path():
    return settings.storage_path(LEARNED_FILE)


def previous_path():
    return settings.storage_path("answer_check_learned.previous.json")


def load(model_name):
    """The learned threshold for `model_name`, or None."""
    try:
        with open(path()) as f:
            learned = json.load(f)
    except (OSError, ValueError):
        return None
    return float(learned["threshold"]) if learned.get("model") == model_name and learned.get("threshold") is not None else None


def score(rows, threshold, cost):
    """+1 per answerable recall answered, -cost per not-answerable recall answered (declined: margin < threshold)."""
    return sum((1.0 if answerable else -cost) for _, margin, answerable in rows if margin >= threshold)


def choose(rows, current, cost):
    """The threshold with the best score on `rows`; ties go to the one nearest `current`."""
    candidates = sorted({m for _, m, _ in rows} | {current})
    return max(candidates, key=lambda t: (score(rows, t, cost), -abs(t - current)))


def run(dry_run=False, force=False):
    from backend.brain.features.answer_check import AnswerChecker
    from backend.brain.memory_ranking import self_improve
    from backend.brain.memory_ranking.ranker import RecallLog
    cfg = config()
    state = self_improve.status()
    entry = {"at": time.time(), "kind": "answer_check", "dry_run": bool(dry_run)}

    def done(**kw):
        entry.update(kw)
        return entry if dry_run else self_improve._record(state, entry)
    if not cfg["enabled"] and not force:
        return {**entry, "status": "disabled"}
    checker = AnswerChecker.from_config()
    if checker is None:
        return {**entry, "status": "disabled", "reason": "the answer check is off or unavailable"}
    if checker.certificate() is not None or (settings.get("recall.answer_check", {}) or {}).get("threshold") is not None:
        return {**entry, "status": "skipped", "reason": "a configured or certified threshold takes precedence"}

    rows = RecallLog().answer_check_feedback(checker.model_name)
    n_pos, n_neg = sum(r[2] for r in rows), sum(not r[2] for r in rows)
    entry.update(answerable=n_pos, not_answerable=n_neg)
    if n_pos < cfg["min_answerable"] or n_neg < cfg["min_unanswerable"]:
        return {**entry, "status": "waiting", "reason": f"{n_pos} answerable and {n_neg} not-answerable recalls with feedback; "
                                                         f"{cfg['min_answerable']} and {cfg['min_unanswerable']} needed"}
    new = len(rows) - int(state.get("answer_check_at_last_attempt", 0))
    if new < cfg["min_new"] and not force:
        return {**entry, "status": "waiting", "reason": f"{new} new recalls with feedback since the last attempt; {cfg['min_new']} needed"}

    share = self_improve.config()["holdout"]
    test = [r for r in rows if self_improve.held_out(r[0], share)]
    train = [r for r in rows if not self_improve.held_out(r[0], share)]
    t_pos, t_neg = sum(r[2] for r in test), sum(not r[2] for r in test)
    if t_pos < cfg["min_holdout_answerable"] or t_neg < cfg["min_holdout_unanswerable"]:
        return {**entry, "status": "waiting", "reason": f"held-out part has {t_pos} answerable and {t_neg} not-answerable recalls; "
                                                         f"{cfg['min_holdout_answerable']} and {cfg['min_holdout_unanswerable']} needed"}
    if not dry_run:
        state["answer_check_at_last_attempt"] = len(rows)
    cost, current = cfg["wrong_answer_cost"], checker.threshold()
    candidate = choose(train, current, cost)
    before, after = score(test, current, cost), score(test, candidate, cost)
    entry.update(current_threshold=round(current, 4), candidate_threshold=round(candidate, 4), wrong_answer_cost=cost,
                 held_out=len(test), held_out_score_current=before, held_out_score_candidate=after)
    if after <= before:
        return done(status="kept_current", reason="the new threshold did not score better on held-out feedback")
    if dry_run:
        return done(status="would_adopt")
    final = choose(rows, current, cost)
    if os.path.exists(path()):
        os.replace(path(), previous_path())
    elif os.path.exists(previous_path()):
        os.remove(previous_path())
    with open(path(), "w") as f:
        json.dump({"model": checker.model_name, "threshold": final, "wrong_answer_cost": cost, "fitted_on": len(rows),
                   "answerable": n_pos, "not_answerable": n_neg, "replaced": current, "at": time.time()}, f, indent=2)
    return done(status="adopted", threshold=round(final, 4))


def rollback():
    """Back to the threshold in use before the last adoption (the default if there was none)."""
    from backend.brain.memory_ranking import self_improve
    state = self_improve.status()
    if os.path.exists(previous_path()):
        os.replace(previous_path(), path())
        what = "previous learned threshold restored"
    elif os.path.exists(path()):
        os.remove(path())
        what = "default threshold restored"
    else:
        return {"kind": "answer_check", "status": "nothing_to_roll_back"}
    return self_improve._record(state, {"at": time.time(), "kind": "answer_check", "status": "rolled_back", "reason": what})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()
    print(json.dumps(rollback() if args.rollback else run(dry_run=args.dry_run, force=True), indent=2))


if __name__ == "__main__":
    main()
