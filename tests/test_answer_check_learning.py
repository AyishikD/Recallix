"""The answer check's decline threshold learned from feedback (backend/brain/features/answer_check_learning.py)."""
import random

import pytest

from backend import settings
from backend.brain.features import answer_check, answer_check_learning as acl
from backend.brain.memory_ranking.ranker import RecallLog


def _rows(n_pos, n_neg, pos_margin, neg_margin, seed=0):
    """(recall_id, margin, answerable) with margins drawn around the given centres."""
    rng = random.Random(seed)
    return ([(f"p{i}", rng.gauss(pos_margin, 1.5), True) for i in range(n_pos)]
            + [(f"n{i}", rng.gauss(neg_margin, 1.5), False) for i in range(n_neg)])


@pytest.fixture
def checker(monkeypatch, tmp_path):
    """A real AnswerChecker without its model; default threshold -2.7 (as shipped); learning on."""
    chk = answer_check.AnswerChecker.__new__(answer_check.AnswerChecker)
    chk.model_name = "m1"
    fitted = tmp_path / "fitted.json"
    fitted.write_text('{"model": "m1", "threshold": -2.7}')
    monkeypatch.setattr(answer_check, "DEFAULT_PATH", str(fitted))
    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: chk))
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda p, d=None: {"answer_check": {"enabled": True}, "holdout": 0.3}
                        if p == "learning" else real(p, d))
    return chk


def _feedback(monkeypatch, rows):
    monkeypatch.setattr(RecallLog, "answer_check_feedback", lambda self, model: rows)


def test_off_by_default():
    assert acl.config()["enabled"] is False
    assert acl.run()["status"] == "disabled"


def test_mostly_answerable_deployment_learns_to_decline_less(checker, monkeypatch):
    # answerable questions whose answer-check margins sit below the shipped threshold, few traps
    _feedback(monkeypatch, _rows(200, 25, pos_margin=-4.0, neg_margin=-9.0))
    assert checker.threshold() == -2.7
    out = acl.run()
    assert out["status"] == "adopted" and out["threshold"] < -2.7
    assert abs(checker.threshold() - out["threshold"]) < 1e-3       # used by every recall from now on
    assert acl.rollback()["status"] == "rolled_back"
    assert checker.threshold() == -2.7                              # back to the default


def test_trap_heavy_deployment_learns_to_decline_more(checker, monkeypatch):
    _feedback(monkeypatch, _rows(100, 80, pos_margin=1.0, neg_margin=-1.5))
    out = acl.run()
    assert out["status"] == "adopted" and out["threshold"] > -2.7


def test_a_wrong_answer_costs_more_than_a_refusal():
    rows = [("a", 1.0, True), ("b", 0.5, False), ("c", 0.0, True)]
    # declining all three scores 0; answering all scores 2 - cost
    assert acl.choose(rows, current=5.0, cost=3.0) > 0.5     # answer only "a": +1
    assert acl.choose(rows, current=5.0, cost=0.5) <= 0.0    # cheap wrong answers: answer all


def test_nothing_changes_without_enough_feedback(checker, monkeypatch):
    _feedback(monkeypatch, _rows(30, 5, pos_margin=-4.0, neg_margin=-9.0))
    assert acl.run()["status"] == "waiting" and checker.threshold() == -2.7


def test_kept_when_not_better_on_held_out_feedback(checker, monkeypatch):
    _feedback(monkeypatch, _rows(120, 40, pos_margin=-2.0, neg_margin=-3.4, seed=3))   # already about right
    out = acl.run()
    assert out["status"] in ("kept_current", "adopted")
    if out["status"] == "kept_current":
        assert checker.threshold() == -2.7


def test_feedback_is_read_verbatim(tmp_path):
    log = RecallLog(str(tmp_path / "log.db"))
    for rid in ("used_checked", "used_unchecked", "none"):
        log.log(rid, "u", [{"id": i, "features": {"semantic": 0.1}} for i in (1, 2, 3)])
        log.log_answer_check(rid, "u", "m1", -1.0, [1, 2])
    log.feedback("used_checked", "u", [2])
    log.feedback("used_unchecked", "u", [3])      # the answer was in memory but the check never read it
    log.feedback("none", "u", [])
    log.feedback("none", "intruder", [1])         # another user's feedback on this recall is ignored
    got = {rid: answerable for rid, _, answerable in log.answer_check_feedback("m1")}
    assert got == {"used_checked": True, "none": False}
