from backend.brain.memory.truth_engine import TruthEngine


def mem(i, content, value, ts, slot="projectalpha::deadline"):
    return {"id": i, "content": content, "timestamp": ts, "relevance": 0.9,
            "metadata": {"slots": [{"slot_id": slot, "value": value}]}}


def resolve(*mems):
    return {m["id"]: m["truth_status"] for m in TruthEngine().resolve(list(mems))}


def test_newest_value_wins():
    assert resolve(mem(1, "deadline is Dec 12", "Dec 12", 1), mem(2, "deadline moved to Dec 18", "Dec 18", 2)) == {2: "verified"}


def test_negating_an_old_value_keeps_the_current_one():
    out = resolve(mem(1, "deadline is December 12", "December 12", 1),
                  mem(2, "deadline was moved to December 18", "December 18", 2),
                  mem(3, "deadline is NOT December 12 anymore", "NOT December 12 anymore", 3))
    assert out == {2: "verified"}


def test_negating_the_current_value_falls_back_to_an_earlier_one():
    out = resolve(mem(1, "deadline is December 12", "December 12", 1),
                  mem(2, "deadline is December 18", "December 18", 2),
                  mem(3, "deadline is not December 18", "not December 18", 3))
    assert out == {1: "verified"}


def test_negating_the_only_value_leaves_nothing():
    assert resolve(mem(1, "deadline is December 12", "December 12", 1),
                   mem(2, "deadline is not December 12", "not December 12", 2)) == {}


def test_a_comma_not_is_not_a_negation():
    out = resolve(mem(1, "My name is Priya", "Priya", 1, "user::name"),
                  mem(2, "My name is Priya, not Pria", "Priya", 2, "user::name"))
    assert out == {2: "verified"}


def test_memories_without_facts_pass_through():
    m = {"id": 9, "content": "Lunch was good.", "timestamp": 1, "relevance": 0.5, "metadata": {}}
    assert resolve(m) == {9: "verified"}
