import pytest

from backend import settings
from backend.brain.features import aggregation_check as ac


def claims(question):
    out = ac.presuppositions(question)
    return out and (out["kind"], out["claims"])


@pytest.mark.parametrize("question, kind, expected", [
    ("How many weddings did I attend?", "counting", ["I did attend weddings."]),
    ("How many times did I actually go to Italy in 2026?", "counting", ["I did go to Italy."]),
    ("How many 5K races have I run?", "counting", ["I have run 5K races."]),
    ("How many books did I finish reading since we started chatting?", "counting", ["I did finish reading books."]),
    ("How many times has John injured his ankle?", "counting", ["John has injured his ankle."]),
    ("How many dogs has Maria adopted from the dog shelter she volunteers at?", "counting",
     ["Maria has adopted dogs from the dog shelter she volunteers at."]),
    ("How many times did I tell you I had started reading a new book?", "counting", ["I had started reading a new book."]),
    ("Which city did I visit first, New York or Boston?", "ordering", ["I did visit New York.", "I did visit Boston."]),
    ("Which came first, my knitting class or my yoga session?", "ordering",
     ["My knitting class happened.", "My yoga session happened."]),
    ("Which came first, ordering the kitchen cabinets or painting the spare bedroom?", "ordering",
     ["I was ordering the kitchen cabinets.", "I was painting the spare bedroom."]),
    ("Which came first, the Foo Fighters show Alex went to or the workshop I attended?", "ordering",
     ["Alex went to the Foo Fighters show.", "I attended the workshop."]),
    ("In what order did I send the coffee table book, the prints, and the flyers?", "ordering",
     ["I did send the coffee table book.", "I did send the prints.", "I did send the flyers."]),
    ("How many days passed between the day I adopted Milo and the day I got my new camera gear?", "time_between",
     ["I adopted Milo.", "I got my new camera gear."]),
    ("How many days after my friend Liza got her haircut did I get a haircut?", "time_between",
     ["My friend Liza got her haircut.", "I did get a haircut."]),
])
def test_questions_become_the_events_they_take_for_granted(question, kind, expected):
    assert claims(question) == (kind, expected)


def test_unclear_phrasings_keep_every_reading():
    # which haircut was "second" is worked out across memories, so the claim is also tried without the ordinal
    _, c = claims("How many weeks passed between my first haircut and my second haircut?")
    assert c == [("My first haircut happened.", "My haircut happened."), ("My second haircut happened.", "My haircut happened.")]
    # "it" may or may not be the thing named before
    _, c = claims("How many days after I signed up for the Lisbon 10K did I actually run it?")
    assert c == ["I signed up for the Lisbon 10K.", ("I did run it.", "I did run the Lisbon 10K.")]
    # later names share "meals with"
    _, c = claims("In what order did I have meals with Maya and Karim?")
    assert c == ["I did have meals with Maya.", ("I did have meals with Karim.", "I did have Karim.")]


@pytest.mark.parametrize("question", [
    "What is my dog's name?", "Where did I go first?", "How many employees does my company have?",
    "How much rent do I pay?", "How many children does Melanie have?", "When did I move to Pune?", "",
])
def test_other_questions_are_left_alone(question):
    assert ac.presuppositions(question) is None


def test_a_statement_about_the_speaker_is_also_read_in_the_first_person():
    assert ac._as_speaker("John has injured his ankle.", "John") == "I have injured my ankle."
    assert ac._as_speaker("Joanna's scripts has been rejected.", "Joanna") == "My scripts has been rejected."
    assert ac._as_speaker("John has injured his ankle.", "Maria") is None


class FakeEntailment:
    """A memory entails a statement iff it contains the statement's key word (its last word, "happened" aside)."""
    model_name = "fake-nli"

    def entailment(self, pairs):
        key = lambda hyp: [w for w in hyp.rstrip(".").split() if w != "happened"][-1].lower()  # noqa: E731
        return [1.0 if key(hyp) in premise.lower() else 0.0 for premise, hyp in pairs]


def test_check_needs_every_event_supported():
    q = "Which came first, my class or my session?"
    memories = [{"id": 1, "content": "Went to my first knitting class today."}, {"id": 2, "content": "Rain all day."}]
    out = ac.check(FakeEntailment(), q, memories, 0.5)
    assert out["kind"] == "ordering" and out["supported"] is False
    assert [(c["statement"], c["support"], c["memory"]) for c in out["claims"]] == \
        [("My class happened.", 1.0, 1), ("My session happened.", 0.0, None)]
    memories.append({"id": 3, "content": "Yoga session at the studio this morning."})
    out = ac.check(FakeEntailment(), q, memories, 0.5)
    assert out["supported"] is True and [c["memory"] for c in out["claims"]] == [1, 3]
    assert ac.check(FakeEntailment(), "What is my dog's name?", memories, 0.5) is None
    assert ac.check(FakeEntailment(), q, [], 0.5) is None


def _store(client, h, text):
    assert client.post("/memory/store", json={"content": text}, headers=h).status_code == 200


def _recall(client, h, query, **extra):
    return client.post("/memory/recall", json={"query": query, **extra}, headers=h).json()["data"]


@pytest.fixture
def aggregation_on(monkeypatch):
    real = settings.get

    def get(key, default=None):
        if key == "recall.aggregation_check":
            return {"enabled": True, "threshold": 0.5, "memories": 10}
        return real(key, default)
    monkeypatch.setattr(settings, "get", get)
    monkeypatch.setattr(ac.EntailmentChecker, "from_config", classmethod(lambda cls: FakeEntailment()))


def test_recall_answers_when_every_event_is_in_memory_and_declines_a_trap(client, api_keys, aggregation_on):
    h = api_keys["alice"]
    _store(client, h, "Flew to Hampi today, the ruins are unreal.")
    _store(client, h, "Reached Gokarna this morning, beach day.")

    out = _recall(client, h, "Which did I visit first, Hampi or Gokarna?")
    assert out["status"] == "CONSISTENT" and len(out["memories"]) == 2
    agg = out["debug"]["aggregation_check"]
    assert agg["kind"] == "ordering" and agg["supported"] and agg["needs_calculation"] and agg["model"] == "fake-nli"
    assert "worked out across the memories" in out["receipt"]["reason"]
    assert out["receipt"]["versions"]["aggregation_check"] == "fake-nli"

    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?")      # Coorg was never visited
    assert out["status"] == "ABSTAINED" and out["memories"] == [] and out["answer"] is None
    assert "I did visit Coorg." in out["receipt"]["reason"] and "no memory supports" in out["receipt"]["reason"]

    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?", keep_on_decline=True)
    assert out["status"] == "ABSTAINED" and len(out["memories"]) == 2


def test_recall_is_unchanged_when_the_check_is_off_or_the_question_is_not_aggregation(client, api_keys, monkeypatch):
    h = api_keys["alice"]
    _store(client, h, "Flew to Hampi today, the ruins are unreal.")
    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?")      # off by default
    assert out["debug"]["aggregation_check"] is None

    real = settings.get
    monkeypatch.setattr(settings, "get", lambda key, default=None: {"enabled": True} if key == "recall.aggregation_check"
                        else real(key, default))
    monkeypatch.setattr(ac.EntailmentChecker, "from_config", classmethod(lambda cls: FakeEntailment()))
    assert _recall(client, h, "Where did I fly today?")["debug"]["aggregation_check"] is None
