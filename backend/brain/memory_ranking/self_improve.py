"""
Learning from use: refits the recall ranker from the feedback this deployment collected, and
keeps the new fit only if it does better on feedback it was not fitted on.

The scheduler runs this every `workers.learning_hours` (config.yaml `learning`). Each run:

  1. takes the recalls that received feedback (POST /memory/feedback). Too few, or too few new
     ones since the last attempt: nothing happens.
  2. sets aside `holdout` of them (chosen by a hash of the recall id, so the same recall is
     always on the same side) and fits a candidate on the rest.
  3. compares the candidate with the model now in use on the held-out recalls. The candidate
     is adopted only if the right memory is ranked first at least as often, is in the first
     five at least as often, one of the two is strictly better, and answerable questions are
     not declined more often than allowed (recall.max_false_abstain_rate) or than now. The
     candidate's decline threshold is the lower of the one fitted and the one the held-out
     recalls allow.
  4. if adopted: the model in use is copied to ranker_model.previous.json, and a fit on all
     the feedback replaces it (picked up without a restart).

Every run is recorded in <storage>/learning_log.json (GET /internal/learning). Only this
deployment's own feedback is used; nothing leaves the server.

    python -m backend.brain.memory_ranking.self_improve              # run once now
    python -m backend.brain.memory_ranking.self_improve --dry-run    # report, change nothing
    python -m backend.brain.memory_ranking.self_improve --rollback   # back to the previous model
    python -m backend.brain.memory_ranking.self_improve --status
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict

from backend import settings
from backend.brain.memory_ranking.fit_ranker import (ANSWER_K, _top_and_hit, fit_model, max_false_abstain_rate, save,
                                                     threshold_for)
from backend.brain.memory_ranking.ranker import Ranker, RecallLog, model_path

HISTORY = 50  # runs kept in the log


def config():
    cfg = settings.get("learning", {}) or {}
    return {"enabled": bool(cfg.get("enabled", True)),
            "min_recalls": int(cfg.get("min_recalls", 30)),      # recalls with feedback before the first fit
            "min_new": int(cfg.get("min_new", 20)),              # new ones before trying again
            "holdout": float(cfg.get("holdout", 0.2)),           # share set aside to judge the candidate
            "min_holdout": int(cfg.get("min_holdout", 6))}       # answerable held-out recalls needed to judge


def log_path():
    return os.path.join(os.path.dirname(model_path()), "learning_log.json")


def previous_path():
    return os.path.join(os.path.dirname(model_path()), "ranker_model.previous.json")


def status():
    try:
        with open(log_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"recalls_at_last_attempt": 0, "runs": []}


def _record(state, entry):
    state["runs"] = (state.get("runs", []) + [entry])[-HISTORY:]
    os.makedirs(os.path.dirname(log_path()), exist_ok=True)
    tmp = log_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, log_path())
    return entry


def held_out(recall_id, share):
    return int(hashlib.sha256(str(recall_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def evaluate(model, queries):
    """Ranking quality of `model` on labelled recalls: answerable ones only (a relevant memory was marked)."""
    first = top_k = declined = n = 0
    for cands in queries:
        if not any(y for _, y in cands):
            continue
        scored = sorted(((Ranker.score(f, model), y) for f, y in cands), key=lambda t: -t[0])
        n += 1
        first += bool(scored[0][1])
        top_k += any(y for _, y in scored[:ANSWER_K])
        declined += scored[0][0] < float(model["threshold"])
    if not n:
        return {"answerable": 0, "first": None, "top_k": None, "wrongly_declined": None}
    return {"answerable": n, "first": round(first / n, 4), "top_k": round(top_k / n, 4),
            "wrongly_declined": round(declined / n, 4)}


def better(candidate, current, allowed_decline):
    """The adoption rule (module docstring, step 3)."""
    return (candidate["first"] >= current["first"] and candidate["top_k"] >= current["top_k"]
            and (candidate["first"] > current["first"] or candidate["top_k"] > current["top_k"])
            and candidate["wrongly_declined"] <= max(current["wrongly_declined"], allowed_decline))


def run(dry_run=False, force=False):
    cfg = config()
    state = status()
    entry = {"at": time.time(), "dry_run": bool(dry_run)}
    done = lambda **kw: entry.update(kw) or (entry if dry_run else _record(state, entry))  # noqa: E731
    if not cfg["enabled"] and not force:
        return {**entry, "status": "disabled"}

    groups = defaultdict(list)
    for recall_id, features, label in RecallLog().labelled():
        groups[recall_id].append((features, label))
    n = len(groups)
    entry["recalls_with_feedback"] = n
    if n < cfg["min_recalls"]:
        return {**entry, "status": "waiting", "reason": f"{n} recalls with feedback; {cfg['min_recalls']} needed"}
    new = n - int(state.get("recalls_at_last_attempt", 0))
    if new < cfg["min_new"] and not force:
        return {**entry, "status": "waiting", "reason": f"{new} new recalls with feedback since the last attempt; {cfg['min_new']} needed"}

    train = [q for rid, q in groups.items() if not held_out(rid, cfg["holdout"])]
    test = [q for rid, q in groups.items() if held_out(rid, cfg["holdout"])]
    current = Ranker.model()
    before = evaluate(current, test)
    if before["answerable"] < cfg["min_holdout"] or not any(y for q in train for _, y in q):
        return {**entry, "status": "waiting",
                "reason": f"{before['answerable']} answerable held-out recalls; {cfg['min_holdout']} needed to judge a new fit"}
    if not dry_run:
        state["recalls_at_last_attempt"] = n

    candidate = fit_model(train, source="feedback (automatic)")
    # The decline threshold was set on the fitted part; on a small held-out part it can decline a
    # little more than allowed by chance. Take the lower of the two, so neither part exceeds the limit.
    candidate["threshold"] = min(candidate["threshold"], threshold_for(
        {"all": [_top_and_hit(q, candidate["weights"]) for q in test if q]}, max_false_abstain_rate()))
    after = evaluate(candidate, test)
    entry.update(held_out_recalls=len(test), fitted_on=len(train), current=before, candidate=after)
    if not better(after, before, max_false_abstain_rate()):
        return done(status="kept_current", reason="the new fit was not better on held-out feedback")
    if dry_run:
        return done(status="would_adopt")

    final = fit_model(list(groups.values()), source="feedback (automatic)")
    final["adopted_after"] = {"held_out_recalls": len(test), "current": before, "candidate": after}
    if os.path.exists(model_path()):
        shutil.copyfile(model_path(), previous_path())
        entry["previous_model"] = previous_path()
    else:
        entry["previous_model"] = "shipped default (delete ranker_model.json to return to it)"
        if os.path.exists(previous_path()):
            os.remove(previous_path())
    os.makedirs(os.path.dirname(model_path()), exist_ok=True)
    save(final)
    return done(status="adopted")


def rollback():
    """Back to the model in use before the last adoption (the shipped default if there was none)."""
    state = status()
    if os.path.exists(previous_path()):
        os.replace(previous_path(), model_path())
        what = "previous model restored"
    elif os.path.exists(model_path()):
        os.remove(model_path())
        what = "shipped default restored"
    else:
        return {"status": "nothing_to_roll_back"}
    return _record(state, {"at": time.time(), "status": "rolled_back", "reason": what})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    out = status() if args.status else rollback() if args.rollback else run(dry_run=args.dry_run, force=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
