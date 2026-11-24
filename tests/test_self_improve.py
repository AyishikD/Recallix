"""Learning from use: automatic ranker refits that are kept only when better on held-out feedback."""
import json
import os
import random

import pytest

from backend.brain.memory_ranking import self_improve
from backend.brain.memory_ranking.ranker import FEATURES, Ranker, RecallLog, model_path


def _feedback(n, relevant, seed=0):
    """n recalls of 8 candidates with random features; `relevant(candidates)` picks the one marked useful."""
    rng = random.Random(seed)
    rows = []
    for r in range(n):
        cands = [{k: 0.0 for k in FEATURES} | {"semantic": rng.random(), "lexical": rng.random(), "recency": rng.random()}
                 for _ in range(8)]
        best = relevant(cands)
        rows += [(f"recall{r}", f, int(f is best)) for f in cands]
    return rows


@pytest.fixture
def feedback(monkeypatch):
    def use(rows):
        monkeypatch.setattr(RecallLog, "labelled", lambda self: rows)
    return use


def test_nothing_happens_until_there_is_enough_feedback(feedback):
    feedback(_feedback(10, lambda c: max(c, key=lambda f: f["lexical"])))
    out = self_improve.run()
    assert out["status"] == "waiting" and "30 needed" in out["reason"]
    assert not os.path.exists(model_path()) or Ranker.model()["source"] != "feedback (automatic)"


def test_a_better_fit_is_adopted_and_can_be_rolled_back(feedback):
    # users keep confirming the memory that shares the most words, whatever its meaning score
    feedback(_feedback(200, lambda c: max(c, key=lambda f: f["lexical"])))
    before = dict(Ranker.model())
    out = self_improve.run()
    assert out["status"] == "adopted", out
    assert out["candidate"]["first"] > out["current"]["first"]
    assert out["candidate"]["wrongly_declined"] <= max(out["current"]["wrongly_declined"], 0.05)
    now = Ranker.model()                                              # picked up without a restart
    assert now["source"] == "feedback (automatic)" and now["weights"]["lexical"] > before["weights"]["lexical"]
    assert self_improve.status()["runs"][-1]["status"] == "adopted"

    assert self_improve.run()["status"] == "waiting"                  # no new feedback since: no second attempt

    assert self_improve.rollback()["status"] == "rolled_back"
    assert Ranker.model()["weights"] == before["weights"]


def test_a_fit_that_is_not_better_on_held_out_feedback_is_not_adopted(feedback):
    current = Ranker.model()
    feedback(_feedback(200, lambda c: max(c, key=lambda f: Ranker.score(f, current))))   # the model in use is already right
    saved = open(model_path()).read() if os.path.exists(model_path()) else None
    out = self_improve.run()
    assert out["status"] == "kept_current", out
    assert out["current"]["first"] == 1.0
    assert (open(model_path()).read() if os.path.exists(model_path()) else None) == saved
    assert json.load(open(self_improve.log_path()))["runs"][-1]["status"] == "kept_current"


def test_dry_run_changes_nothing(feedback):
    feedback(_feedback(200, lambda c: max(c, key=lambda f: f["lexical"])))
    before = Ranker.model()["weights"]
    assert self_improve.run(dry_run=True)["status"] == "would_adopt"
    assert Ranker.model()["weights"] == before and not os.path.exists(self_improve.log_path())


def test_the_scheduler_runs_it_as_a_job(feedback):
    from backend.api.deps import get_scheduler
    feedback([])
    assert get_scheduler().run_job("learning")["status"] == "waiting"
