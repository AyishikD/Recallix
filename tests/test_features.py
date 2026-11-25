from backend.brain.features.facts import extract_triads, to_slots
from backend.brain.features.query import (analyze_query, capitalised_spans, memory_subject, resolve_first_person,
                                          spans_match)
from backend.brain.memory.recall_engine import RecallEngine


def slots(text):
    return {s["slot_id"]: s["value"] for s in to_slots(extract_triads(text))}


def test_spans_match_inside_longer_names_but_not_different_ids():
    assert spans_match(("x99",), ("galaxy", "x99", "primary", "emission"))
    assert spans_match(("kestrel",), ("project", "kestrel"))
    assert not spans_match(("galaxy", "x1"), ("galaxy", "x2"))
    assert not spans_match(("falconet",), ("falcon",))


def test_sentence_initial_word_is_not_a_question_entity():
    assert analyze_query("Status of X-77?")["strong_spans"] == [("x77",)]
    assert ("caroline",) in analyze_query("What did Caroline realize?")["strong_spans"]


def test_vocatives_do_not_make_a_memory_about_someone():
    owner, mentions = memory_subject("[8 May] Melanie: Thanks, Caroline! I ran a race.", speaker="Melanie")
    assert owner == ("melanie",) and ("caroline",) not in mentions
    owner, _ = memory_subject("I live in Pune.")
    assert owner == ("user",)
    assert capitalised_spans("Wow Jon, same here!", skip_vocatives=True) == []


def test_first_person_resolved_to_recorded_speaker_only():
    assert resolve_first_person("[d] Melanie: I'm tired and my dog is sick.", "Melanie") == \
        "[d] Melanie: Melanie is tired and Melanie's dog is sick."
    assert resolve_first_person("I live in Pune.", None) == "I live in Pune."


def test_general_fact_forms():
    assert slots("Starship Daedalus reactor core is CRITICAL.") == {"starshipdaedalus::reactor_core": "CRITICAL"}
    assert slots("Patient X-100 Status: Critical condition.") == {"patientx100::status": "Critical condition"}
    assert slots("ALERT: Galaxy X-55 has shifted. Primary emission is now Inverse Compton Scattering.") == \
        {"galaxyx55::primary_emission": "Inverse Compton Scattering"}
    assert slots("Galaxy X-77 is colloquially known as 'The Dragon's Breath'.") == {"dragonsbreath::alias_of": "Galaxy X-77"}
    assert slots("The Project Alpha deadline was moved to December 18.") == {"projectalpha::deadline": "December 18"}
    # precision guard: casual statements are not facts
    assert slots("My cat is sleeping on the couch.") == {}
    assert slots("Primary emission is now unknown.") == {}  # no subject to carry


def test_alias_rewrite():
    assert RecallEngine._replace_alias("What powers 'The Dragon's Breath'?", "dragonsbreath", "Galaxy X-77") == \
        "What powers Galaxy X-77?"
    assert RecallEngine._replace_alias("What OS does Falcon run?", "falcon", "staging server") == \
        "What OS does staging server run?"
    assert RecallEngine._replace_alias("Nothing here", "falcon", "x") is None


def test_texting_drops_the_subject_but_it_is_still_about_the_user():
    assert slots("update: moved to pune last month for the new job") == {"user::location": "pune"}
    assert slots("uff moved to pune last month, new job at infosys hinjewadi")["user::location"] == "pune"
    assert slots("allergic to penicillin, found out the hard way lol") == {"user::allergy": "penicillin"}
    assert slots("working at infosys since june") == {"user::employer": "infosys"}
    assert slots("I moved to Pune last month.") == {"user::location": "Pune"}       # time words are not the place
    assert "user::location" not in slots("Priya moved to Pune")                     # third person stays third person
    assert "user::location" not in slots("moved the couch to the bedroom")
