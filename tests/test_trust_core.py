"""
Trust core: provenance, determinism (replay), memory tests, calibrated refusals, receipts.
All hermetic: sandboxes use temporary storage and an in-process exact engine.
"""
import json
import math
import random
import textwrap

import pytest


# --------------------------------------------------------------------------- calibration
def test_binomial_cdf_matches_exact_sum():
    from backend.brain.features.calibration import binom_cdf
    exact = sum(math.comb(10, k) * 0.3 ** k * 0.7 ** (10 - k) for k in range(3))
    assert abs(binom_cdf(2, 10, 0.3) - exact) < 1e-12
    assert binom_cdf(-1, 10, 0.3) == 0.0 and binom_cdf(10, 10, 0.3) == 1.0


def test_too_little_calibration_data_never_declines():
    from backend.brain.features.calibration import certify_threshold, min_calibration_size
    assert min_calibration_size(0.05, 0.05) == 59
    cert = certify_threshold([1.0] * 30, alpha=0.05, delta=0.05)
    assert not cert["certified"] and cert["threshold"] is None


def test_guarantee_holds_in_simulation():
    """The certified threshold's true wrong-refusal rate may exceed alpha in at most ~delta of draws."""
    from backend.brain.features.calibration import certify_threshold
    rng = random.Random(7)
    trials, violations = 200, 0
    for _ in range(trials):
        cert = certify_threshold([rng.gauss(0, 1) for _ in range(250)], alpha=0.05, delta=0.05)
        t = cert["threshold"]
        true_rate = 0.0 if t is None else 0.5 * (1 + math.erf(t / math.sqrt(2)))
        violations += true_rate > 0.05
    assert violations / trials <= 0.08  # delta = 0.05 plus Monte Carlo slack


def test_certificate_drives_the_threshold(monkeypatch):
    from backend.brain.features import answer_check, calibration
    cert = calibration.certify_threshold([float(i) for i in range(200)], 0.05, 0.05)
    calibration.save(cert, "fake-model")
    checker = answer_check.AnswerChecker.__new__(answer_check.AnswerChecker)
    checker.model_name = "fake-model"
    monkeypatch.setattr(answer_check, "_cfg", lambda: {"threshold": None})
    assert checker.threshold() == cert["threshold"]
    calibration.save(calibration.certify_threshold([1.0] * 10), "fake-model")  # not enough data
    assert checker.threshold() == float("-inf")


# --------------------------------------------------------------------------- provenance
def test_facts_cite_their_exact_source():
    from backend.brain.features.facts import EXTRACTOR_VERSION, extract_triads, to_slots
    msg = "Hi. My name is Priya and I live in Pune. The Atlas launch moved to July 14."
    slots = {s["slot_id"]: s for s in to_slots(extract_triads(msg), timestamp=100.0)}
    # a run-on value no longer hides the next fact in the sentence
    assert slots["user::name"]["value"] == "Priya" and slots["user::location"]["value"] == "Pune"
    for s in slots.values():
        a, b = s["source"]["span"]
        assert msg[a:b] == s["value"]
        assert s["source"]["rule"] and s["source"]["extractor"] == EXTRACTOR_VERSION
        assert s["version"] == 100_000  # from the memory's timestamp, not the wall clock


# --------------------------------------------------------------------------- replay / determinism
MESSAGES = [
    "My name is Priya and I live in Pune.",
    "I'm allergic to peanuts.",
    "The Atlas launch moved to July 14.",
    "Project Alpha deadline is December 12.",
    "The deadline was moved to December 18.",
    "The weather is nice today.",
]


def _seed(box, user="alice"):
    with box.silenced():
        for i, m in enumerate(MESSAGES):
            box.mm.store(user, m, timestamp=1_700_000_000 + i)


def test_replay_is_deterministic_and_reproduces_the_store():
    from backend.tools import replay
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        _seed(box)
        original = replay.snapshot(box.mm.episodic.db_path)
        rows = replay._rows(box.mm.episodic.db_path)
    first = replay.replay(rows, embedder="hash")
    second = replay.replay(rows, embedder="hash")
    assert replay.diff(first, second) == []           # same inputs -> identical memory
    assert replay.diff(original, first) == []         # replay reproduces what was stored
    assert len(first["memories"]) == len(MESSAGES)


def test_diff_reports_changed_facts():
    from backend.tools import replay
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        _seed(box)
        a = replay.snapshot(box.mm.episodic.db_path)
    b = json.loads(json.dumps(a))
    b["memories"][0]["facts"] = [f for f in b["memories"][0]["facts"] if f["fact"] != "user::location"]
    lines = replay.diff(a, b)
    assert any("fact removed user::location='Pune'" in line for line in lines)


def test_sandbox_never_touches_production_storage(tmp_path):
    from backend import settings
    from backend.tools.sandbox import Sandbox
    before = str(settings.storage_root())
    with Sandbox(embedder="hash") as box:
        assert str(settings.storage_root()) != before
        _seed(box)
    assert str(settings.storage_root()) == before


# --------------------------------------------------------------------------- memory tests
def test_memtest_passes_and_fails_correctly(tmp_path):
    from backend.tools import memtest
    good = tmp_path / "good.yaml"
    good.write_text(textwrap.dedent("""
        name: remembers where Priya lives
        store:
          - "My name is Priya and I live in Pune."
          - "The weather is nice today."
        expect:
          - fact: user::location
            contains: Pune
          - recall: "Where do I live?"
            top_contains: Pune
    """))
    bad = tmp_path / "bad.yaml"
    bad.write_text(textwrap.dedent("""
        name: wrong expectation must fail
        store: ["My name is Priya and I live in Pune."]
        expect:
          - fact: user::location
            equals: Mumbai
    """))
    assert memtest.main([str(good), "--embedder", "hash", "--answer-check", "off"]) == 0
    assert memtest.main([str(bad), "--embedder", "hash", "--answer-check", "off"]) == 1


# --------------------------------------------------------------------------- receipts
def test_recall_comes_with_a_receipt():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash", answer_check=False) as box:
        _seed(box)
        with box.silenced():
            out = box.mm.retrieve("alice", "Where do I live?")
    r = out["receipt"]
    assert r["decision"] == out["status"] and r["reason"]
    assert r["versions"]["extractor"].startswith("rules-")
    assert out["memories"], "expected a memory"
    top = out["memories"][0]["receipt"]
    assert top["memory_id"] == out["memories"][0]["id"] and top["stored_at"] == 1_700_000_000
    assert {"fact": "user::location", "value": "Pune"}.items() <= next(
        f for f in top["facts"] if f["fact"] == "user::location").items()
    assert next(f for f in top["facts"] if f["fact"] == "user::location")["quote"] == "Pune"


def test_novelty_uses_exact_search():
    """Importance must not depend on the approximate graph (timing, random level seed)."""
    import numpy as np
    import backend.brain.memory.semantic.semantic_memory as sm
    seen = {}

    def fake_post(path, body=None):
        seen.update(body or {})
        return {"results": [{"id": "1", "score": 0.5}]}
    original = sm.engine_post
    sm.engine_post = fake_post
    try:
        assert sm.SemanticMemory.nearest_similarity(object.__new__(sm.SemanticMemory), "u", np.ones(4)) == 0.5
    finally:
        sm.engine_post = original
    assert seen.get("exact") is True


# --------------------------------------------------------------------------- coreference across messages
def test_the_deadline_resolves_to_the_earlier_entity():
    from backend.brain.features.facts import extract_triads, resolve_definite, to_slots
    slots = to_slots(extract_triads("The deadline was moved to December 18."), timestamp=2.0)
    resolved = resolve_definite(slots, lambda rel: "projectalpha" if rel == "deadline" else None)
    assert [s["slot_id"] for s in resolved] == ["projectalpha::deadline"]
    assert resolved[0]["source"]["coreference"] == {"phrase": "the deadline", "resolved_to": "projectalpha"}
    # nothing earlier to refer to: left as is
    assert resolve_definite(slots, lambda rel: None)[0]["slot_id"] == "deadline::when"


def test_coreference_respects_private_memories():
    """Another agent's private fact must not be used to interpret this agent's message."""
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        with box.silenced():
            box.mm.store("u", "Project Alpha deadline is December 12.", agent_id="a1", memory_type="private", timestamp=1.0)
            box.mm.store("u", "The deadline was moved to December 18.", agent_id="a2", memory_type="private", timestamp=2.0)
        assert box.mm.episodic.latest_entity_for_relation("u", "deadline", "a2", 3.0) is None
        assert box.mm.episodic.latest_entity_for_relation("u", "deadline", "a1", 3.0) == "projectalpha"


# --------------------------------------------------------------------------- early exit
class CountingChecker:
    """Answers iff the context contains `needle`; counts how many memories it read."""
    model_name = "counting"

    def __init__(self, needle, threshold=0.0):
        self.needle, self._t, self.reads = needle, threshold, 0

    def threshold(self):
        return self._t

    def check(self, question, contexts):
        self.reads += len(contexts)
        return [{"margin": 1.0 if self.needle in c else -1.0, "answer": self.needle if self.needle in c else ""}
                for c in contexts]


def test_early_exit_reads_less_and_decides_the_same():
    from backend.brain.memory.recall_engine import RecallEngine
    mems = [{"id": i, "content": c} for i, c in enumerate(
        ["I live in Pune.", "The weather is nice.", "I like tea.", "My dog is Pixel.", "I work at Acme."])]

    found = CountingChecker("Pune")
    read, best, answered = RecallEngine._read_until_answer(found, "Where do I live?", [dict(m) for m in mems], 0.0)
    assert answered and best == 1.0 and found.reads == 1 and len(read) == 1      # answer at rank 1: one read

    later = CountingChecker("Pixel")
    read, best, answered = RecallEngine._read_until_answer(later, "Dog's name?", [dict(m) for m in mems], 0.0)
    assert best == 1.0 and later.reads == 5 and [m["id"] for m in read] == [0, 1, 2, 3, 4]  # 1 + 2 + 2

    absent = CountingChecker("Mumbai")
    read, best, answered = RecallEngine._read_until_answer(absent, "Where?", [dict(m) for m in mems], 0.0)
    assert not answered and best == -1.0 and absent.reads == 5       # a decline reads everything

    calibrating = CountingChecker("Pune")                             # measure mode: never stop early
    RecallEngine._read_until_answer(calibrating, "Where?", [dict(m) for m in mems], -1e9)
    assert calibrating.reads == 5


# --------------------------------------------------------------------------- who said what
class PersonAwareChecker:
    """Like a real reader: finds 'peanuts' strongly only when the asked person is the one in the memory."""
    model_name = "person-aware"

    def threshold(self):
        return 0.0

    def check(self, question, contexts):
        out = []
        for c in contexts:
            asked = [n for n in ("Alice", "Bob") if n in question]
            strong = "peanuts" in c and any(n in c for n in asked)
            out.append({"margin": 2.0 if strong else (0.5 if "peanuts" in c else -1.0), "answer": "peanuts" if "peanuts" in c else ""})
        return out


def test_subject_contrast_rejects_memories_about_someone_else(monkeypatch):
    from backend import settings
    from backend.brain.memory.recall_engine import RecallEngine
    real_get = settings.get
    monkeypatch.setattr(settings, "get", lambda p, d=None: {"subject_contrast": {"enabled": True}}
                        if p == "recall.answer_check" else real_get(p, d))
    alice = {"id": 1, "content": "I'm allergic to peanuts.", "metadata": {"client": {"speaker": "Alice"}}}
    bob = {"id": 2, "content": "I love hiking.", "metadata": {"client": {"speaker": "Bob"}}}
    checker = PersonAwareChecker()

    accept, subject = RecallEngine._subject_contrast(checker, "What is Bob allergic to?", [alice, bob])
    assert subject == "Bob"
    read, best, answered = RecallEngine._read_until_answer(checker, "What is Bob allergic to?", [dict(alice)], 0.0, accept)
    assert best > 0 and not answered            # the reader "found" peanuts, but it is Alice's allergy
    assert read[0]["subject_contrast"] <= 0

    accept, subject = RecallEngine._subject_contrast(checker, "What is Alice allergic to?", [alice, bob])
    read, best, answered = RecallEngine._read_until_answer(checker, "What is Alice allergic to?", [dict(alice)], 0.0, accept)
    assert answered and read[0]["subject_contrast"] > 0

    # no named speaker in the question: the check does not apply
    assert RecallEngine._subject_contrast(checker, "What am I allergic to?", [alice, bob]) == (None, None)


# --------------------------------------------------------------------------- fact half-lives
def test_freshness_learns_from_the_users_update_history():
    from backend.brain.memory import freshness
    day = freshness.DAY
    now = 400 * day
    history = [  # status changes often; location once
        ("user::status", "status", "busy", 0.0), ("user::status", "status", "free", 10 * day),
        ("user::status", "status", "busy", 20 * day), ("user::status", "status", "free", 30 * day),
        ("user::location", "location", "Pune", 0.0), ("user::location", "location", "Pune", 200 * day),  # restated
    ]
    rates = freshness.fit_rates(history, now, a0=1.0, b0_days=365.0)
    assert rates["status"]["events"] == 3 and rates["location"]["events"] == 0
    assert rates["status"]["half_life_days"] < rates["location"]["half_life_days"]
    runs = freshness._runs(history)
    fresh = freshness.assess("user::location", "location", "Pune", runs, rates, now)
    stale = freshness.assess("user::status", "status", "busy", runs, rates, now)
    current = freshness.assess("user::status", "status", "free", runs, rates, now)
    assert stale["superseded"] and stale["still_true"] == 0.0 and stale["replaced_by"] == "free"
    assert fresh["since"] == 200 * day                       # restating renewed it
    assert 0 < current["still_true"] < fresh["still_true"] <= 1


def test_recall_receipt_reports_freshness():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash", answer_check=False) as box:
        _seed(box)
        with box.silenced():
            out = box.mm.retrieve("alice", "Where do I live?")
    fact = next(f for m in out["memories"] for f in m["receipt"]["facts"] if f["fact"] == "user::location")
    assert fact["freshness"]["superseded"] is False and 0 < fact["freshness"]["still_true"] <= 1


# --------------------------------------------------------------------------- recall-time distillation
def test_confirmed_answers_are_remembered_for_similar_questions():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash", answer_check=False) as box:
        _seed(box)
        with box.silenced():
            first = box.mm.retrieve("alice", "Where do I live?")
        target = next(m["id"] for m in first["memories"] if "Pune" in m["content"])
        assert box.mm.recall_engine.confirm(first["recall_id"], "alice", [target]) > 0
        with box.silenced():
            again = box.mm.retrieve("alice", "Where do I live?")
        hit = next(m for m in again["memories"] if m["id"] == target)
        assert hit["receipt"]["confirmed_for"] == "Where do I live?"
        # another agent cannot see alice's private memory through a confirmation
        with box.silenced():
            other = box.mm.retrieve("alice", "Where do I live?", agent_id="someone_else")
        assert all(m["id"] != target for m in other["memories"])
        # deleting the memory removes it from future recalls even though it was confirmed
        box.mm.delete("alice", target)
        with box.silenced():
            after = box.mm.retrieve("alice", "Where do I live?")
        assert all(m["id"] != target for m in after["memories"])
