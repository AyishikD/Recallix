import pytest

from backend import settings
from backend.brain.features import aggregation_check as ac


def claims(question):
    """The rewriting under test, without the "did" readings every "-ing" statement also gets (tested below)."""
    out = ac.presuppositions(question)
    if not out:
        return out
    def strip(claim):
        if not isinstance(claim, tuple):
            return claim
        extra = {d for v in claim for d in ac._did_form(v)}
        kept = tuple(v for v in claim if v not in extra)
        return kept[0] if len(kept) == 1 else kept
    return out["kind"], [strip(c) for c in out["claims"]]


def test_an_ing_event_is_also_read_with_did():
    assert ac._did_form("I was signing the Camry paperwork.") == ("I did sign the Camry paperwork.", "I did signe the Camry paperwork.")
    assert ac._did_form("I was shopping for a sofa.") == ("I did shop for a sofa.",)
    assert ac._did_form("My knitting class happened.") == ()


@pytest.mark.parametrize("memory, plan", [
    ("booked flights to goa for march!!", True),
    ("signed up for the city marathon!", True),
    ("planning to visit hampi next week", True),
    ("Booked and paid for the Bali villa today, flying out on 3 June", True),
    ("My next big race is the Lisbon Half Marathon on March 5!", True),
    ("Flew to Bali today!", False),
    ("Just got back from Hampi, unreal", False),
    ("Unboxed my new camera gear, can't wait to test it on my next shoot!", False),
    ("I will never forget the wedding", False),
])
def test_plan_only_memories(memory, plan):
    assert ac.plan_only(memory) is plan


def test_a_plan_does_not_prove_the_event_happened():
    memories = [{"id": 1, "content": "booked flights to goa for march"}, {"id": 2, "content": "the trip is off, cancelled everything"}]
    out = ac.check(FakeEntailment(), "How many times did I fly to goa?", memories, 0.5)
    assert out["supported"] is False and out["claims"][0]["plan_only"]["memory"] == 1
    # a question about the plan itself may stand on it
    out = ac.check(FakeEntailment(), "How many days after I booked the goa flights did I cancel?", memories, 0.5)
    assert out["claims"][0]["memory"] == 1


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
     [("I was ordering the kitchen cabinets.", "Ordering the kitchen cabinets happened."),
      ("I was painting the spare bedroom.", "Painting the spare bedroom happened.")]),
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


@pytest.mark.parametrize("question, kind, expected", [
    ("Baby showers - how many have I actually attended?", "counting", ["I have attended baby showers."]),
    ("What's the number of weddings I've been to?", "counting", ["I have been to weddings."]),
    ("Count the in-person workshops I went to, please.", "counting", ["I went to in-person workshops."]),
    ("How often did I make a dessert?", "counting", ["I did make a dessert."]),
    ("Italy - how many times have I gone there?", "counting", ["I have gone to Italy."]),
    ("Getting a flu shot - how many times have I done that?", "counting", ["I was getting a flu shot."]),
    ("For my sister, how many birthday gifts did I buy?", "counting", ["I did buy birthday gifts for my sister."]),
    ("'Deep Work' and 'Atomic Habits' - put them in the order I finished them.", "ordering",
     ["I finished 'Deep Work'.", "I finished 'Atomic Habits'."]),
    ("The Mazda MX-5 or the Hyundai Elantra - which car did I own earlier?", "ordering",
     ["I did own the Mazda MX-5.", "I did own the Hyundai Elantra."]),
    ("My yoga session or my knitting class - which one came first?", "ordering",
     ["My yoga session happened.", "My knitting class happened."]),
    ("What was the order in which I installed the smart thermostat and the smart kitchen scale?", "ordering",
     ["I installed the smart thermostat.", "I installed the smart kitchen scale."]),
    ("What was the order of me clearing the attic and cleaning the gutters?", "ordering",
     [("I was clearing the attic.", "Clearing the attic happened."),
      ("I was cleaning the gutters.", "Cleaning the gutters happened.")]),
    ("Did I have the MRI scan before or after starting the infusions, and which was earlier?", "ordering",
     ["I did have the MRI scan.", ("I was starting the infusions.", "Starting the infusions happened.")]),
    ("home reno check - how many internal walls did I demolish?", "counting", ["I did demolish internal walls."]),
    ("Count the separate occasions on which I installed insulation in the house.", "counting",
     ["I installed insulation in the house."]),
    ("What was the gap between sending the preprint to my examiners and my oral defense?", "time_between",
     [("I was sending the preprint to my examiners.", "Sending the preprint to my examiners happened."),
      ("I was sending my oral defense.", "Sending my oral defense happened.", "My oral defense happened.")]),
    ("Which was shipped first by me: the wedding album or the photo set?", "ordering",
     ["I shipped the wedding album.", "I shipped the photo set."]),
    ("I signed the lease - how many days later did I host a housewarming party?", "time_between",
     ["I signed the lease.", "I did host a housewarming party."]),
    ("From the day I signed the lease to the day I moved in - how many days was it?", "time_between",
     ["I signed the lease.", "I moved in."]),
    ("In weeks and days, how long was it between the day I bought the grill and the day I got back from Pune?", "time_between",
     ["I bought the grill.", "I got back from Pune."]),
    ("My school announced the snow day - how many days after I backed up my photos was that?", "time_between",
     ["I backed up my photos.", "My school announced the snow day."]),
])
def test_freer_phrasings_are_understood_too(question, kind, expected):
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
    "How often does John take his kids to the park?", "How long has Melanie been practicing art?",
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
            return {"enabled": True, "mode": "decide", "threshold": 0.5, "memories": 10}
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

    _store(client, h, "Booked a homestay in Coorg for March!")             # a plan is not a visit
    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?")
    assert out["status"] == "ABSTAINED" and "only records a plan" in out["receipt"]["reason"]

    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?", keep_on_decline=True)
    assert out["status"] == "ABSTAINED" and len(out["memories"]) == 3


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


class NoAnswerReader:
    """An answer check that never finds an answer phrase, so the usual path declines."""
    model_name = "no-answer"

    def threshold(self):
        return 0.0

    def check(self, question, contexts):
        return [{"margin": -1.0, "answer": ""} for _ in contexts]


class AlwaysAnswerReader(NoAnswerReader):
    model_name = "always"

    def check(self, question, contexts):
        return [{"margin": 1.0, "answer": c.split()[0] if c.split() else ""} for c in contexts]


@pytest.fixture
def rescue_mode(monkeypatch):
    real = settings.get

    def get(key, default=None):
        if key == "recall.aggregation_check":
            return {"enabled": True, "mode": "rescue", "threshold": 0.5, "memories": 10}
        if key == "recall.answer_check":
            return {**(real(key, default) or {}), "calculation_questions": False}
        return real(key, default)
    monkeypatch.setattr(settings, "get", get)
    monkeypatch.setattr(ac.EntailmentChecker, "from_config", classmethod(lambda cls: FakeEntailment()))


def test_rescue_answers_a_supported_question_the_usual_path_declined(client, api_keys, rescue_mode, monkeypatch):
    from backend.brain.features import answer_check
    h = api_keys["alice"]
    _store(client, h, "Flew to Hampi today, the ruins are unreal.")
    _store(client, h, "Reached Gokarna this morning, beach day.")
    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: NoAnswerReader()))
    out = _recall(client, h, "Which did I visit first, Hampi or Gokarna?")
    assert out["status"] == "CONSISTENT" and len(out["memories"]) == 2
    assert out["debug"]["aggregation_check"]["rescued"] is True
    assert out["receipt"]["reason"].startswith("the answer check found no answer phrase, but the memories support")
    # an event no memory supports: the decline stands
    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?")
    assert out["status"] == "ABSTAINED" and out["memories"] == []


def test_rescue_never_declines_what_the_usual_path_answered(client, api_keys, rescue_mode, monkeypatch):
    from backend.brain.features import answer_check
    h = api_keys["alice"]
    _store(client, h, "Flew to Hampi today, the ruins are unreal.")
    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: AlwaysAnswerReader()))
    out = _recall(client, h, "Which did I visit first, Hampi or Coorg?")   # Coorg unsupported, but not its call
    assert out["status"] == "CONSISTENT" and out["debug"]["aggregation_check"] is None


def test_mode_defaults_to_rescue_and_rejects_unknown_values(monkeypatch):
    real = settings.get
    for given, expected in ((None, "rescue"), ("decide", "decide"), ("DECIDE", "decide"), ("veto", "rescue")):
        cfg = {"enabled": True} if given is None else {"enabled": True, "mode": given}
        monkeypatch.setattr(settings, "get", lambda key, default=None, cfg=cfg: cfg if key == "recall.aggregation_check"
                            else real(key, default))
        assert ac.config()["mode"] == expected


def _mem(i, day):
    from datetime import datetime, timezone
    return {"id": i, "timestamp": datetime(2025, 1, 1, 12, tzinfo=timezone.utc).timestamp() + (day - 1) * 86400}


def _claims(*items, kind="time_between"):
    return {"kind": kind, "supported": True,
            "claims": [{"statement": s, "support": 0.99, "memory": m, "supporters": sup, **({"ordinal": o} if o else {})}
                       for s, m, sup, o in items]}


def test_computed_days_between_two_verified_events():
    mems = {1: _mem(1, 10), 2: _mem(2, 51)}
    out = ac.compute(_claims(("I signed the lease.", 1, {1: 0.99}, None), ("I moved in.", 2, {2: 0.98}, None)), mems)
    assert out["kind"] == "time_between" and out["text"] == "41 days (5 weeks 6 days)"
    assert "2025-01-10 (memory 1) to 2025-02-20 (memory 2) = 41 days" == out["working"]


def test_no_computed_answer_when_the_events_are_uncertain():
    mems = {1: _mem(1, 10), 2: _mem(2, 51), 3: _mem(3, 70)}
    lease, moved = ("I signed the lease.", 1, {1: 0.99}, None), ("I moved in.", 2, {2: 0.98}, None)
    # the second event dated before the first
    assert ac.compute(_claims(("I moved in.", 2, {2: 0.98}, None), lease), mems) is None
    # an event supported on two different dates, and the question does not say which
    assert ac.compute(_claims(lease, ("I moved in.", 2, {2: 0.98, 3: 0.97}, None)), mems) is None
    # weak support, counting, or a check that did not pass
    assert ac.compute(_claims(lease, ("I moved in.", 2, {2: 0.6}, None)), mems) is None
    assert ac.compute({**_claims(lease, moved), "kind": "counting"}, mems) is None
    assert ac.compute({**_claims(lease, moved), "supported": False}, mems) is None


def test_first_and_last_choose_among_several_dates():
    mems = {1: _mem(1, 5), 2: _mem(2, 30), 3: _mem(3, 60)}
    out = ac.compute(_claims(("My first gym visit happened.", 2, {2: 0.99, 1: 0.95}, "first"),
                             ("I did the bench press.", 3, {3: 0.99}, None)), mems)
    assert out["text"].startswith("55 days") and out["events"][0]["memory"] == 1


def test_computed_order_and_same_day_events():
    mems = {1: _mem(1, 40), 2: _mem(2, 10), 3: _mem(3, 10)}
    out = ac.compute(_claims(("I visited Hampi.", 1, {1: 0.99}, None), ("I visited Gokarna.", 2, {2: 0.99}, None),
                             kind="ordering"), mems)
    assert out["text"] == "I visited Gokarna -> I visited Hampi"
    assert ac.compute(_claims(("I visited Gokarna.", 2, {2: 0.99}, None), ("I visited Coorg.", 3, {3: 0.99}, None),
                              kind="ordering"), mems) is None
