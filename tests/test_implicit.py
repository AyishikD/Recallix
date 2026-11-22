"""Implicit facts: text helpers and recall wiring (models replaced by stand-ins)."""
from backend.brain.features import implicit


def test_instantiate_names_people_and_conjugates():
    assert implicit.instantiate("gets a driver's licence", "xEffect", "Priya", None) == "Priya gets a driver's licence."
    assert implicit.instantiate("get degree certificate", "xEffect", "Priya", None) == "Priya gets degree certificate."
    assert implicit.instantiate("gets married", "oEffect", "Priya", "Priya's sister") == "Priya's sister gets married."
    assert implicit.instantiate("PersonX attends PersonY's wedding", "isBefore", "Priya", "Sam") == "Priya attends Sam's wedding."
    assert implicit.instantiate("INFIDELITY", "xEffect", "Priya", None) is None          # noise
    assert implicit.instantiate("gets married", "oEffect", "Priya", None) is None         # "others": too vague
    assert implicit.instantiate("PersonY smiles", "xEffect", "Priya", None) is None       # needs a second person


def test_people_finds_kin_and_names():
    found = implicit.people("Priya just got back from Priya's sister's wedding with Sam.", "Priya")
    assert "Priya's sister" in found and "Sam" in found and "Priya" not in found


class StandInInference:
    source = "test"

    def infer(self, text, speaker=None, min_entailment=0.9, min_similarity=0.55):
        if "driving test" in text:
            return [{"statement": f"{speaker or 'I'} gets a driver's licence.", "entailment": 0.99,
                     "pattern": "PersonX passes PersonX's driving test", "relation": "xEffect", "similarity": 0.8}]
        return []


class LicenceReader:
    model_name = "licence-reader"

    def threshold(self):
        return 0.0

    def check(self, question, contexts):
        return [{"margin": 2.0 if "licence" in c else -2.0, "answer": "a driver's licence" if "licence" in c else ""}
                for c in contexts]


def test_question_time_inference_answers_with_its_reasoning(monkeypatch):
    from backend.brain.memory.recall_engine import RecallEngine
    monkeypatch.setattr(implicit.ImplicitInference, "from_config", classmethod(lambda cls: StandInInference()))
    mems = [{"id": 1, "content": "I finally passed my driving test today!", "metadata": {"client": {"speaker": "Priya"}}},
            {"id": 2, "content": "The weather is nice.", "metadata": {}}]
    hit = RecallEngine._answer_by_inference(LicenceReader(), "Does Priya have a licence?", mems, 0.0)
    assert hit["id"] == 1 and hit["answer"] == "a driver's licence"
    assert hit["inferred"]["statement"] == "Priya gets a driver's licence." and hit["inferred"]["relation"] == "xEffect"
    assert RecallEngine._answer_by_inference(LicenceReader(), "Q?", mems[1:], 0.0) is None


def test_disabled_by_default():
    assert implicit.rules_config()["enabled"] is False
    assert implicit.ImplicitInference.from_config() is None
