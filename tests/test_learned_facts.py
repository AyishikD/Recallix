"""Learned fact extractor: merging with rule facts, entity attribution, multi-valued relations."""
from backend.brain.features import facts, learned_facts


class StandInExtractor:
    version = "model-test-000"

    def extract(self, text):
        out = []
        for value, rel in (("Pune", "location"), ("cycling", "hobby_cycling"), ("2015 Honda Civic", "vehicle")):
            i = text.find(value)
            if i >= 0:
                out.append({"entity": learned_facts.entity_for(text, i), "relation": rel, "value": value,
                            "rule": self.version, "extractor": self.version, "span": [i, i + len(value)]})
        return out


def test_rules_first_model_fills_gaps():
    msg = "I live in Pune and I love cycling on weekends. I drive a 2015 Honda Civic."
    rule_triads = facts.extract_triads(msg)
    merged = facts.with_learned(msg, rule_triads, StandInExtractor())
    by_value = {t["value"]: t for t in merged}
    assert by_value["Pune"]["rule"] == "first_person"          # the rule fact wins; no duplicate
    assert [t["value"] for t in merged].count("Pune") == 1
    assert by_value["cycling"]["rule"] == "model-test-000"     # gap filled by the model
    slots = {s["slot_id"]: s for s in facts.to_slots(merged, timestamp=1.0)}
    assert slots["user::vehicle"]["source"]["extractor"] == "model-test-000"
    assert facts.with_learned(msg, rule_triads, None) == rule_triads


def test_entity_attribution():
    text = "My sister lives in Toronto and Sam works at Google, while I live in Pune."
    assert learned_facts.entity_for(text, text.index("Toronto")) == "sister"
    assert learned_facts.entity_for(text, text.index("Google")) == "Sam"
    assert learned_facts.entity_for(text, text.index("Pune")) == "user"


def test_off_by_default():
    assert learned_facts.config()["enabled"] is False and learned_facts.LearnedExtractor.from_config() is None
