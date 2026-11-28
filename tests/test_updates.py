"""Updates beat originals, and third-person questions about the user (backend/brain/memory/updates.py)."""
from backend.brain.memory.updates import promote_updates, user_perspective


def _ranked(rows):
    """rows: [(id, content, timestamp, relevance, semantic)] in rank order."""
    return [{"id": i, "content": c, "timestamp": t, "relevance": r, "semantic_score": s} for i, c, t, r, s in rows]


def test_a_later_update_outranks_the_original():
    ranked = _ranked([(1, "meera has ballet on tuesdays 4pm at indiranagar club", 1, 0.5, 0.7),
                      (2, "update: ballet got rescheduled to wednesday afternoons", 4, 0.1, 0.6),
                      (3, "school reopens monday", 0, 0.05, 0.2)])
    out = promote_updates(ranked, "What day does Meera have ballet now?")
    assert [m["id"] for m in out[:2]] == [2, 1] and out[0]["update_of"] == 1


def test_an_update_about_another_named_thing_does_not_replace_the_answer():
    ranked = _ranked([(1, "Project Beta deadline is December 21.", 1, 0.6, 0.6),
                      (2, "The Project Alpha deadline was moved to December 18.", 5, 0.5, 0.55)])
    assert [m["id"] for m in promote_updates(ranked, "When is Project Beta due?")] == [1, 2]


def test_an_update_of_another_attribute_does_not_replace_the_answer():
    ranked = _ranked([(1, "The Kestrel team lead is Marco.", 1, 0.6, 0.6),
                      (2, "Project Kestrel deadline moved to May 24.", 5, 0.5, 0.5)])
    assert [m["id"] for m in promote_updates(ranked, "Who leads Kestrel?")] == [1, 2]


def test_an_update_of_another_part_of_the_message_is_not_promoted():
    ranked = _ranked([(1, "priya asked me to cover her night duty on 14th, taking mom to the eye clinic that morning", 1, 0.3, 0.44),
                      (2, "priya begged again so agreed for night shift on 28th instead", 9, 0.1, 0.08)])
    assert [m["id"] for m in promote_updates(ranked, "Where did I take my mother for an eye checkup?")] == [1, 2]


def test_a_negation_alone_is_not_an_update():
    ranked = _ranked([(1, "The Alpha deadline was moved to December 18.", 3, 0.6, 0.6),
                      (2, "The Alpha deadline is NOT December 12 anymore.", 5, 0.5, 0.6)])
    assert [m["id"] for m in promote_updates(ranked, "What is the latest Alpha deadline?")] == [1, 2]


def test_third_person_questions_about_the_user():
    assert user_perspective("Which hospital does she work at?") == "Which hospital do I work at?"
    assert user_perspective("Where did she take her mother?") == "Where did I take my mother?"
    assert user_perspective("What is the user's job?") == "What is my job?"
    assert user_perspective("Where is the user based?") == "Where am I based?"
    assert user_perspective("Which branch of SBI handles his pension?") == "Which branch of SBI handles my pension?"
    # unchanged: the pronoun refers to someone named earlier, or there is none
    assert user_perspective("When did Meera say she would come?") == "When did Meera say she would come?"
    assert user_perspective("Where does Karthik live?") == "Where does Karthik live?"
    assert user_perspective("Where do I live?") == "Where do I live?"
    q = "What would be a good hobby related to his travel dreams for Tim to pick up?"   # LoCoMo: Tim is a speaker
    assert user_perspective(q, is_speaker=lambda n: n == "Tim") == q


def test_recall_reports_the_rewrite_and_finds_the_answer():
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder="hash") as box:
        with box.silenced():
            box.mm.store("u", "uff shift just started at apollo greams road hospital", timestamp=1.0)
            box.mm.store("u", "karthik works at a bank in dallas", timestamp=2.0)
            out = box.mm.retrieve("u", "Which hospital does she work at?")
        assert out["debug"]["read_as"] == "Which hospital do I work at?"
        assert "apollo" in out["memories"][0]["content"]


def test_purchase_changes_are_updates_but_a_measurement_change_is_not():
    ranked = _ranked([(1, "bought second hand winter jacket canada goose from marketplace", 1, 0.23, 0.5),
                      (2, "canada goose jacket zipper broken, returned to seller and bought columbia parka", 5, 0.15, 0.4)])
    assert [m["id"] for m in promote_updates(ranked, "What winter coat did I end up buying?")] == [2, 1]
    # Gemini batch 3: a later lab value does not replace the prescription
    ranked = _ranked([(1, "serum ferritin is at 14 ng/ml so doc started me on ferrex 150 capsules", 1, 0.5, 0.6),
                      (2, "ferritin level bumped up to 38 ng/ml on the 6-week recheck", 5, 0.4, 0.5)])
    assert [m["id"] for m in promote_updates(ranked, "What iron supplement was prescribed?")] == [1, 2]


def test_reader_sees_texting_with_sentence_case_and_cased_text_unchanged():
    from backend.brain.features.answer_check import restore_case
    assert restore_case("moving to bansko instead, i love it") == "Moving to bansko instead, I love it."
    assert restore_case("blume term sheet signed!! huge relief yar") == "Blume term sheet signed!! Huge relief yar."
    for text in ("Melanie: I went to the park.", "saw Doc Miller", "", "12:30"):
        assert restore_case(text) == text
    s = "take celebrex 200mg daily"
    assert restore_case(s).lower().rstrip(".") == s   # case-only: answer offsets still index the original


def _answered(rows):
    """rows: [(id, content, timestamp, answer)] in served order, all found answering by the reader."""
    return [{"id": i, "content": c, "timestamp": t, "answer": a, "_answer_ok": True, "relevance": 0.5,
             "semantic_score": 0.5} for i, c, t, a in rows]


def test_disagreeing_answers_serve_the_later_one_or_the_earlier_when_asked():
    from backend.brain.memory.updates import resolve_answer_conflicts
    served = _answered([(1, "landlord at my studio on boulevard saint-germain raised the charges", 1, "boulevard saint-germain"),
                        (2, "found a small chambre de bonne studio on rue de la roquette near bastille", 5, "rue de la roquette")])
    assert [m["id"] for m in resolve_answer_conflicts(served, "Where is the new studio located?")] == [2, 1]
    served = _answered([(2, "forearm burn healed, no need for antibiotic ointment anymore", 5, "antibiotic ointment"),
                        (1, "burned my forearm, applied silver sulfadiazine cream", 1, "silver sulfadiazine cream")])
    assert [m["id"] for m in resolve_answer_conflicts(served, "What ointment was originally applied to the forearm burn?")] == [1, 2]


def test_no_conflict_when_the_later_answer_is_already_in_the_earlier_memory():
    from backend.brain.memory.updates import resolve_answer_conflicts
    served = _answered([(1, "serum ferritin is at 14 ng/ml so doc started me on ferrex 150 capsules", 1, "ferrex"),
                        (2, "ferritin level bumped up to 38 ng/ml on the recheck", 5, "ferritin")])
    assert [m["id"] for m in resolve_answer_conflicts(served, "What iron supplement was prescribed?")] == [1, 2]


def test_the_reader_does_not_see_other_peoples_clauses_unless_asked_about_them():
    from backend.brain.memory.updates import without_other_people as w
    roommate = "roommate liam failed his organic chemistry midterm, he is locking himself in the library"
    assert w(roommate, "What major is the athlete studying?") == ""
    assert w(roommate, "What did my roommate fail?") == roommate
    assert w(roommate, "What did Liam fail?") == roommate
    sister = "my sister chloe is moving to austin next month because her husband marcus got a job at tesla"
    assert w(sister, "Where is Chloe moving?") == "my sister chloe is moving to austin next month"
    own = "told priya no chance, taking mom to shankara nethralaya for her followup"
    assert w(own, "Where did I go that morning?") == own   # mom is not the subject
    assert w("Melanie: I went to the park with my kids.", "What did Melanie do?") == "Melanie: I went to the park with my kids."


def test_answers_that_are_not_statements_do_not_count():
    from backend.brain.memory.recall_engine import RecallEngine
    says_nothing = RecallEngine._answer_says_nothing
    people = ["Caroline", "Melanie"]
    assert says_nothing("Melanie", "[3 May] Caroline: Thanks, Melanie! You play any instruments?", people)  # greeting
    assert says_nothing("Caroline", "Melanie: Wow, Caroline, that looks amazing! What inspired it?", people)
    assert says_nothing("Did you make it", "Caroline: That bowl is gorgeous! Did you make it?", people, "Caroline")
    # an assistant's suggestion phrased as a question is an answer when no person is asked about
    assert not says_nothing("Kyoto", "Assistant: Have you considered Kyoto in spring?", ["User", "Assistant"])
    assert not says_nothing("Melanie", "Caroline: I went hiking with Melanie last week.", people)
    assert not says_nothing("pottery", "Melanie: I took a pottery class last Friday.", people)
    assert not says_nothing("looks great", "Melanie: Wow, that agency looks great! What made you pick it?", people)
    assert not says_nothing("blume", "term sheet signed with blume", [])  # texting without punctuation


def test_the_asked_persons_message_to_the_listener_is_about_the_listener(monkeypatch):
    from backend.brain.memory import recall_engine
    monkeypatch.setattr(recall_engine, "memory_speaker", lambda m: m["speaker"])
    about = recall_engine.RecallEngine._about_the_listener
    reaction = {"speaker": "Caroline", "content": "Caroline: Your charity race sounds great! Raising awareness for mental health is huge."}
    assert about(reaction, "Caroline")                                   # asked about Caroline: this is about Mel
    assert not about(reaction, "Melanie")                                # asked about Mel: "your" is Mel
    assert not about({"speaker": "Caroline", "content": "Caroline: I ran it for mental health, you'd love it"}, "Caroline")
    assert not about(reaction, None)                                     # no single person named in the question


def test_best_answer_is_the_most_confident_and_reports_a_later_one():
    from backend.brain.memory.recall_engine import RecallEngine
    final = [{"_answer_ok": True, "answer": "Mumbai", "answer_margin": 2.0, "answer_conflict_with": [1]},
             {"_answer_ok": True, "answer": "Pune", "answer_margin": 6.0},
             {"_answer_ok": False, "answer": "Goa", "answer_margin": 9.0}]
    out = [{"id": 2}, {"id": 1}, {"id": 3}]
    best = RecallEngine._best_answer(final, out)
    assert best["text"] == "Pune" and best["memory_id"] == 1               # not the rejected "Goa"
    assert best["later_answer"] == "Mumbai" and best["later_memory_id"] == 2
    assert RecallEngine._best_answer([{"_answer_ok": False, "answer": "x"}], [{"id": 1}]) is None


# --- the top-level answer never comes from a memory a later message updated ---------------------------------

def test_an_update_that_already_ranks_first_is_still_marked_as_the_update():
    ranked = _ranked([(2, "update: ballet got rescheduled to wednesdays", 4, 0.6, 0.6),
                      (1, "my daughter meera has ballet on tuesdays 4pm", 1, 0.5, 0.7)])
    out = promote_updates(ranked, "When is Meera's ballet?")
    assert [m["id"] for m in out] == [2, 1]  # no move needed
    assert out[0]["update_of"] == 1           # but recorded, so the old answer is not shown


def test_an_update_about_another_named_thing_is_not_marked_even_when_it_ranks_first():
    ranked = _ranked([(2, "The Project Alpha deadline was moved to December 18.", 5, 0.6, 0.6),
                      (1, "Project Beta deadline is December 21.", 1, 0.5, 0.55)])
    out = promote_updates(ranked, "When is Project Beta due?")
    assert out[0].get("update_of") is None


def _accepted(id_, answer, margin, **extra):
    return {"id": id_, "answer": answer, "answer_margin": margin, "_answer_ok": True, **extra}


def test_the_answer_skips_a_memory_a_later_message_updated():
    from backend.brain.memory.recall_engine import RecallEngine
    final = [_accepted(2, "wednesdays", 3.6, update_of=1), _accepted(1, "tuesdays 4pm", 10.1)]
    best = RecallEngine._best_answer(final, [{"id": 2}, {"id": 1}])
    assert best["text"] == "wednesdays" and best["memory_id"] == 2  # not the more confident, outdated span


def test_no_answer_rather_than_an_outdated_one():
    from backend.brain.memory.recall_engine import RecallEngine
    final = [{"id": 2, "update_of": 1, "answer": None, "_answer_ok": False}, _accepted(1, "tuesdays 4pm", 10.1)]
    assert RecallEngine._best_answer(final, [{"id": 2}, {"id": 1}]) is None  # "ballet is cancelled": no new day


def test_without_updates_the_most_confident_answer_is_unchanged():
    from backend.brain.memory.recall_engine import RecallEngine
    final = [_accepted(2, "fridays", 1.0), _accepted(1, "tuesdays 4pm", 10.1)]
    assert RecallEngine._best_answer(final, [{"id": 2}, {"id": 1}])["text"] == "tuesdays 4pm"


# --- a first-person question about now takes the later of two different answers ---------------------------

def test_which_questions_ask_about_the_users_present():
    from backend.brain.memory.updates import asks_about_now
    for q in ("Who is my manager?", "Where do I live?", "What car do I drive?", "Who is my dentist now?"):
        assert asks_about_now(q), q
    for q in ("Who was my first manager?", "What did I eat yesterday?", "What is Caroline excited about?",
              "Where does Melanie live?", "When was my appointment originally?", "Have I ever been to Goa?"):
        assert not asks_about_now(q), q


def test_the_later_answer_wins_for_my_current_state_and_the_confident_one_otherwise():
    from backend.brain.memory.recall_engine import RecallEngine
    final = [_accepted(2, "rohan", 2.4, answer_conflict_with=[1]), _accepted(1, "priya", 7.6)]
    mems = [{"id": 2}, {"id": 1}]
    assert RecallEngine._best_answer(final, mems, "Who is my manager?")["text"] == "rohan"
    third_person = RecallEngine._best_answer(final, mems, "Who is Caroline's manager?")
    assert third_person["text"] == "priya" and third_person["later_answer"] == "rohan"   # unchanged behaviour
