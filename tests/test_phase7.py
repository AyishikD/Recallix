"""Phase 7: conflict alerts, verifiable deletion (more tests are added as features land)."""
import sqlite3


def _store(box, user, text, ts, **kw):
    with box.silenced():
        return box.mm.store(user, text, timestamp=ts, **kw)


# --------------------------------------------------------------------------- conflict alerts
def test_store_flags_a_contradiction_but_keeps_the_newest_value():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        first = _store(box, "u", "I live in Pune.", 1.0)
        assert first["conflicts"] == []
        same = _store(box, "u", "Just to say, I live in Pune.", 2.0)
        assert same["conflicts"] == []                         # restating the same value is not a conflict
        moved = _store(box, "u", "I live in Mumbai now.", 3.0)
        c = moved["conflicts"][0]
        assert c["fact"] == "user::location" and c["previous_value"] == "Pune" and c["new_value"].startswith("Mumbai")
        assert c["previous_memory_id"] == same["memory_id"] and c["previous_said_at"] == 2.0
        with box.silenced():
            out = box.mm.retrieve("u", "Where do I live?")
        assert "Mumbai" in out["memories"][0]["content"]       # the newest value still wins


def test_conflicts_respect_private_memories():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        _store(box, "u", "I live in Pune.", 1.0, agent_id="a1", memory_type="private")
        other = _store(box, "u", "I live in Mumbai.", 2.0, agent_id="a2", memory_type="private")
        assert other["conflicts"] == []                        # a2 never saw a1's private fact


# --------------------------------------------------------------------------- verifiable deletion
def test_delete_returns_a_receipt_and_leaves_nothing_behind():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        res = _store(box, "u", "My name is Priya and I live in Pune.", 1.0)
        mid = res["memory_id"]
        with box.silenced():
            out = box.mm.retrieve("u", "Where do I live?")
        box.mm.recall_engine.confirm(out["recall_id"], "u", [mid])
        receipt = box.mm.delete("u", mid)
        assert receipt["all_copies_gone"] and all(receipt["verified_absent"].values())
        assert receipt["removed"]["memory"] == 1 and receipt["removed"]["confirmed_answers"] == 1
        assert receipt["removed"]["ranking_features"] >= 1 and receipt["removed"]["vector"] == 1
        assert receipt["removed"]["trust_log"] == 1            # the recall's logged trust for this memory
        assert len(receipt["content_sha256"]) == 64 and "Priya" not in str(receipt)  # no content in the log
        assert box.mm.delete("u", mid) is None                 # already gone
        assert box.mm.audit.receipts("u")[0]["memory_id"] == mid
        assert box.mm.audit.receipts("someone-else") == []


def test_deletion_log_detects_tampering():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        for i, text in enumerate(["I live in Pune.", "I like tea.", "My dog is Pixel."]):
            box.mm.delete("u", _store(box, "u", text, float(i))["memory_id"])
        ok, n, bad = box.mm.audit.verify()
        assert ok and n == 3 and bad is None
        with sqlite3.connect(box.mm.audit.db_path) as c:   # someone edits an old entry
            c.execute("UPDATE deletions SET receipt = replace(receipt, 'user request', 'never happened') WHERE idx = 2")
        ok, n, bad = box.mm.audit.verify()
        assert not ok and bad == 2


def test_delete_api_returns_the_receipt(client, api_keys):
    h = api_keys["alice"]
    mid = client.post("/memory/store", json={"content": "I live in Pune."}, headers=h).json()["data"]["memory_id"]
    r = client.delete(f"/memory/{mid}", headers=h).json()["data"]
    assert r["deleted"] == mid and r["receipt"]["all_copies_gone"]
    assert client.get("/memory/deletions", headers=h).json()["data"][0]["memory_id"] == mid
    assert client.get("/memory/deletions", headers=api_keys["bob"]).json()["data"] == []


# --------------------------------------------------------------------------- unknown person
def test_question_about_someone_not_in_memory_is_declined():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash", answer_check=False) as box:
        _store(box, "u", "I work at Zomato as a data analyst.", 1.0)
        _store(box, "u", "Anita lives in Bangalore.", 2.0)
        with box.silenced():
            out = box.mm.retrieve("u", "Where does Morgan work?")
            anita = box.mm.retrieve("u", "Where does Anita live?")
        assert out["status"] == "ABSTAINED" and "Morgan" in out["receipt"]["reason"]
        assert anita["status"] == "CONSISTENT" and "Bangalore" in anita["memories"][0]["content"]


# --------------------------------------------------------------------------- fuzzer
def test_fuzzer_generates_and_checks_cases():
    from backend.tools import fuzz
    msgs = [{"text": t} for t in ["My name is Priya.", "I live in Pune.", "I'm allergic to peanuts.",
                                  "Project Alpha deadline is December 12."]]
    rep = fuzz.fuzz(msgs, embedder="hash")
    kinds = {c["kind"] for c in rep["cases"]}
    assert kinds == {"ask", "someone", "change"} and rep["facts"] >= 4
    assert rep["summary"]["someone"]["pass_rate"] == 1.0          # never hands the user's facts to a stranger
    change = [c for c in rep["cases"] if c["kind"] == "change"]
    assert all(c["conflict_flagged"] for c in change)            # every update raised a conflict alert
    failing = {**rep, "cases": [{**rep["cases"][0], "passed": False}]}
    spec = fuzz.failures_as_memtests(msgs, failing)[0]
    assert spec["store"] and spec["expect"]                      # failures become runnable memtests


# --------------------------------------------------------------------------- trust score
def test_trust_score_is_a_probability_and_can_be_disabled(monkeypatch):
    from backend import settings
    from backend.brain.features import trust
    strong = {"relevance": 0.95, "answer_margin": 6.0, "answer": "peanuts", "receipt": {"facts": []}}
    weak = {"relevance": 0.05, "answer_margin": -8.0, "answer": "", "receipt": {"facts": []}}
    s, w = trust.score(strong), trust.score(weak)
    assert s is not None and 0 <= w < s <= 1                     # fitted model shipped: trust.default.json
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda p, d=None: {"enabled": False} if p == "recall.trust" else real(p, d))
    assert trust.score(strong) is None


def test_recalibration_recovers_a_known_scale_and_shift():
    import random
    from backend.brain.features import trust
    rng = random.Random(0)
    zs = [rng.uniform(-4, 4) for _ in range(3000)]
    ys = [int(rng.random() < trust.recalibrated(z, 0.6, 1.2)) for z in zs]  # the score is under-confident
    a, b = trust.fit_recalibration(zs, ys)
    assert abs(a - 0.6) < 0.1 and abs(b - 1.2) < 0.15
    assert trust.fit_recalibration([], []) == (1.0, 0.0)       # no labels: no change


def test_feedback_becomes_trust_calibration_data_and_receipts_say_which_calibration():
    from backend.tools.sandbox import Sandbox
    from backend.brain.features import trust
    with Sandbox(embedder="hash") as box:
        mid = _store(box, "u", "I'm allergic to peanuts.", 1.0)["memory_id"]
        with box.silenced():
            out = box.mm.retrieve("u", "What am I allergic to?")
        assert out["receipt"]["trust_calibration"]["source"].startswith("reference")
        box.mm.recall_engine.confirm(out["recall_id"], "u", [mid])
        samples = box.mm.recall_engine.recall_log.trust_samples()
        assert len(samples) == 1 and samples[0][1] == 1
        before = out["memories"][0]["trust"]
        trust.save_calibration(1.0, 2.0, {"n": 60, "fitted_at": "test"})
        with box.silenced():
            again = box.mm.retrieve("u", "What am I allergic to?")
        assert again["receipt"]["trust_calibration"] == {"source": "this deployment", "n": 60, "fitted_at": "test"}
        assert again["memories"][0]["trust"] > before          # shifted up by the deployment's correction


# --------------------------------------------------------------------------- proactive recall
def test_proactive_recall_surfaces_what_matters_with_a_reason():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        _store(box, "u", "I'm allergic to peanuts.", 1.0)
        _store(box, "u", "I live in Pune.", 2.0)
        _store(box, "u", "My diet is vegetarian.", 3.0, agent_id="other", memory_type="private")
        hits = box.mm.proactive("u", "Booking a table at the Thai place tonight")
        assert [h["value"] for h in hits] == ["peanuts"] and "eating or ordering food" in hits[0]["why"]
        assert box.mm.proactive("u", "The weather is nice today.") == []
        assert any(h["value"] == "vegetarian" for h in box.mm.proactive("u", "Ordering pizza", agent_id="other"))
