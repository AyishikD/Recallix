"""recall.second_pass (places after the first five) and recall.answer_check.calculation_questions."""
import pytest

from backend import settings
from backend.brain.memory.recall_engine import RecallEngine

T0 = 1_700_000_000.0


@pytest.fixture
def override(monkeypatch):
    """override({"recall.second_pass": {...}}) -> those settings for this test, the rest unchanged."""
    real = settings.get

    def use(values):
        monkeypatch.setattr(settings, "get", lambda path, default=None: values[path] if path in values else real(path, default))
    return use


def _box():
    from backend.tools.sandbox import Sandbox
    return Sandbox(embedder="hash")


CHAT = ["My sister Priya is visiting from Delhi this weekend",
        "We are taking her to the lake on Saturday",
        "The dog chewed my running shoes again",
        "I booked the vet for the dog on Tuesday morning",
        "She said the limp is probably just a sprain",
        "Work has been busy since the product launch",
        "The launch went well and the team got a bonus",
        "I want to learn to bake sourdough bread",
        "My neighbour gave me some sourdough starter",
        "The starter needs feeding every day",
        "I planted tomatoes and basil on the balcony",
        "The tomatoes need more sun than the balcony gets"]


def test_neighbours_are_nearest_first_and_stay_inside_the_conversation():
    with _box() as box:
        with box.silenced():
            ids = [box.mm.store("u", text, timestamp=T0 + i)["memory_id"] for i, text in enumerate(CHAT[:5])]
            later = box.mm.store("u", CHAT[5], timestamp=T0 + 3 * 3600)["memory_id"]   # three hours on: another conversation
        near = [m["id"] for m in box.mm.episodic.neighbours("u", ids[2], window=2)]
        assert near == [ids[3], ids[1], ids[4], ids[0]]                 # next, previous, second next, second previous
        assert [m["id"] for m in box.mm.episodic.neighbours("u", ids[4], window=2)] == [ids[3], ids[2]]   # not `later`
        assert box.mm.episodic.neighbours("u", later, window=2) == []
        assert box.mm.episodic.neighbours("someone_else", ids[2], window=2) == []


def test_second_pass_keeps_the_first_results_and_adds_the_message_next_to_the_best_one(override):
    question = "When is the vet appointment for the dog?"
    with _box() as box:
        with box.silenced():
            for i, text in enumerate(CHAT):
                box.mm.store("u", text, timestamp=T0 + i)
            override({"recall.second_pass": {"enabled": False}})
            before = box.mm.retrieve("u", question, limit=10)["memories"]
            assert not any(m.get("found_by") for m in before)
            override({"recall.second_pass": {"enabled": True, "keep": 2, "search": 1, "neighbours": 1}})
            after = box.mm.retrieve("u", question, limit=10)["memories"]
    assert [m["id"] for m in after[:2]] == [m["id"] for m in before[:2]]          # the first results do not move
    assert sorted(m["id"] for m in after) == sorted(m["id"] for m in before) or len(after) == 10   # nothing dropped
    assert after[0]["content"] == CHAT[3]
    assert after[2]["found_by"] == "second_search"
    near = [m for m in after if m.get("found_by") == "neighbour"]
    assert near and all(m["neighbour_of"] == after[0]["id"] for m in near)
    assert {m["content"] for m in near} <= {CHAT[2], CHAT[4]}                     # the lines around the vet booking
    assert "She said the limp is probably just a sprain" in [m["content"] for m in after[:5]]


def test_second_pass_needs_room_after_the_kept_results(override):
    with _box() as box:
        with box.silenced():
            for i, text in enumerate(CHAT):
                box.mm.store("u", text, timestamp=T0 + i)
            override({"recall.second_pass": {"enabled": True, "keep": 5}})
            out = box.mm.retrieve("u", "When is the vet appointment for the dog?", limit=5)["memories"]
    assert not any(m.get("found_by") for m in out)                                 # limit <= keep: nothing to rearrange


def test_calculation_questions_are_recognised_only_when_enabled(override):
    asks = RecallEngine._needs_calculation
    override({"recall.answer_check": {"calculation_questions": False}})
    assert not asks("How many times did we plan to hike together?")
    override({"recall.answer_check": {"calculation_questions": True}})
    for q in ("How many times did we plan to hike together?", "How long has it been since I adopted Toby?",
              "How many months passed between adopting Toby and Buddy?", "How long ago was my 18th birthday?"):
        assert asks(q), q
    for q in ("Where did Audrey get Pixie from?", "Who knows how to bake?", "What is my dog's name?",
              "How many total employees work at the rescue shelter?", "How much rent does she pay?"):   # single facts
        assert not asks(q), q


class _WordOverlap:
    """Stand-in cross-encoder: scores a memory by the question words it contains."""
    model_name = "test"

    def scores(self, query, texts):
        q = {w.strip("?.,").lower() for w in query.split() if len(w) > 3}
        return [float(len(q & {w.strip("?.,").lower() for w in t.split()})) for t in texts]


def test_rerank_reorders_only_after_the_kept_results(override, monkeypatch):
    from backend.brain.features.reranker import Reranker
    question = "Which balcony plants need more sun?"
    with _box() as box:
        with box.silenced():
            for i, text in enumerate(CHAT):
                box.mm.store("u", text, timestamp=T0 + i * 7200)        # hours apart: no neighbours
            override({"recall.second_pass": {"enabled": False}, "recall.rerank": {"enabled": False}})
            before = box.mm.retrieve("u", question, limit=8)["memories"]
            monkeypatch.setattr(Reranker, "from_config", classmethod(lambda cls, force=False: _WordOverlap()))
            override({"recall.second_pass": {"enabled": False},
                      "recall.rerank": {"enabled": True, "keep": 2, "candidates": 50}})
            after = box.mm.retrieve("u", question, limit=8)["memories"]
    assert [m["id"] for m in after[:2]] == [m["id"] for m in before[:2]]
    rest = [m["rerank_score"] for m in after[2:]]
    assert rest == sorted(rest, reverse=True) and all("rerank_score" not in m for m in after[:2])


def test_rerank_is_off_by_default_and_a_missing_model_changes_nothing(override, monkeypatch):
    from backend.brain.features import reranker
    assert reranker.config()["enabled"] is False
    monkeypatch.setattr(reranker.Reranker, "from_config", classmethod(lambda cls, force=False: None))   # e.g. no download possible
    with _box() as box:
        with box.silenced():
            for i, text in enumerate(CHAT):
                box.mm.store("u", text, timestamp=T0 + i * 7200)
            override({"recall.second_pass": {"enabled": False}, "recall.rerank": {"enabled": False}})
            plain = [m["id"] for m in box.mm.retrieve("u", "Which balcony plants need more sun?", limit=8)["memories"]]
            override({"recall.second_pass": {"enabled": False}, "recall.rerank": {"enabled": True, "keep": 2}})
            same = [m["id"] for m in box.mm.retrieve("u", "Which balcony plants need more sun?", limit=8)["memories"]]
    assert plain == same


def test_deep_recall_runs_the_reranker_only_when_asked(override, monkeypatch):
    from backend.brain.features.reranker import Reranker
    monkeypatch.setattr(Reranker, "from_config", classmethod(lambda cls, force=False: _WordOverlap()))
    question = "Which balcony plants need more sun?"
    with _box() as box:
        with box.silenced():
            for i, text in enumerate(CHAT):
                box.mm.store("u", text, timestamp=T0 + i * 7200)
            override({"recall.second_pass": {"enabled": False}, "recall.rerank": {"enabled": False, "keep": 2}})
            plain = box.mm.retrieve("u", question, limit=8)["memories"]
            deep = box.mm.retrieve("u", question, limit=8, deep=True)["memories"]
            override({"recall.second_pass": {"enabled": False}, "recall.rerank": {"enabled": True, "keep": 2}})
            not_deep = box.mm.retrieve("u", question, limit=8, deep=False)["memories"]
    assert not any("rerank_score" in m for m in plain + not_deep)       # off by config, or switched off for this recall
    assert any("rerank_score" in m for m in deep)
    assert [m["id"] for m in deep[:2]] == [m["id"] for m in plain[:2]]


def test_deep_flag_reaches_the_engine_through_the_api(monkeypatch):
    from backend.api.public import public_api
    seen = {}

    class FakeManager:
        def retrieve(self, user_id, query, agent_id="default_agent", limit=5, return_candidates=False, deep=None,
                     keep_on_decline=None):
            seen["deep"], seen["keep"] = deep, keep_on_decline
            return {"status": "NO_MATCH", "memories": []}
    monkeypatch.setattr(public_api, "get_memory_manager", lambda: FakeManager())
    public_api.recall_memory(public_api.RecallRequest(query="q", deep=True), auth={"user_id": "u", "role": "user"})
    assert seen["deep"] is True and seen["keep"] is None
    public_api.recall_memory(public_api.RecallRequest(query="q", keep_on_decline=True), auth={"user_id": "u", "role": "user"})
    assert seen["keep"] is True


def test_spread_out_questions_are_recognised_for_automatic_deep_recall():
    needs = RecallEngine._needs_deep
    assert needs("What kinds of classes has Audrey joined?")
    assert needs("What activities has Melanie done with her kids?")
    assert needs("What do Audrey and Andrew have in common?")
    assert not needs("Where do I live?")
    assert not needs("When did she move to Pune?")
