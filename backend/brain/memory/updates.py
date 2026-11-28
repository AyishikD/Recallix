"""
Two LLM-free recall steps that do not depend on fact extraction (so they also work for
casual, lowercase texting, where the fact grammar rarely fires):

1. promote_updates — "updates beat originals".
   When a memory ranks above a LATER memory that announces a change ("rescheduled to",
   "instead", "update:", "moving meera from fitso to dolfin", "cancelled", ...) and the two
   are about the same thing (they share a specific word, e.g. "ballet" or "revotronics"),
   the later one is ranked just above the earlier one. Fact slots already arbitrate this
   when facts are extracted (TruthEngine); this covers the messages no rule understood.

2. user_perspective — questions about the user asked in the third person.
   Agents often ask "Which hospital does she work at?" or "What is the user's job?" while
   memories are written as "I ...". Unless the question names someone before the pronoun
   (then the pronoun is about that person), he/she/they and "the user" are read as I/my.

Both are deterministic and can be switched off (recall.updates.enabled,
recall.user_perspective.enabled).
"""
import re
from functools import lru_cache

from backend import settings


_CHANGE_CUE = re.compile(
    r"\b(?:instead|rescheduled|re-scheduled|postponed|preponed|pushed (?:back|to)|moved (?:to|from|it)|"
    r"moving \w+ (?:from|to)|shift(?:ed|ing) (?:to|out|from)|switch(?:ed|ing) (?:to|from)|changed (?:to|it)|"
    r"update[d:]?|replac(?:ed|ing)|new (?:date|time|place|address)|now (?:at|in|on|with|it'?s)|"
    # (A "quantity changed to a number" cue, "dropped retainer to 4000", was tried on Gemini batch 2 and
    # failed unseen batch 3: "ferritin bumped up to 38" outranked the prescription it did not replace.)
    # a purchase undone and replaced: "returned to seller and bought columbia parka", "exchanged it for"
    r"return(?:ed|ing)\b[^.]{0,60}?\b(?:bought|got|ordered|picked up)|exchang(?:ed|ing)\s+(?:\w+\s+){0,2}for)", re.I)
# Only replacement cues: a pure negation ("not X anymore", "cancelled") says what is no longer true,
# not what replaced it, so it must not outrank the message that does (the fact layer handles negations).
_WORD = re.compile(r"[a-z0-9][a-z0-9'’-]*")
_GENERIC = {"yar", "bhai", "today", "tomorrow", "yesterday", "next", "week", "month", "time", "day", "days", "got",
            "get", "going", "gonna", "will", "just", "now", "also", "really", "still", "again", "back", "new", "old",
            "one", "ones", "thing", "things", "said", "told", "asked", "make", "made", "take", "taking", "took",
            "good", "great", "bad", "lol", "haha", "okay", "finally", "actually", "instead", "update", "updated",
            "moved", "moving", "cancelled", "canceled", "rescheduled", "dropped", "anymore", "longer", "since",
            "user", "assistant"}


def _stopwords():
    from backend.brain.memory_ranking.importance_ranker import STOPWORDS
    return STOPWORDS


def _stem(w):
    """Crude and deterministic: leads ~ lead, classes ~ class, internship ~ intern."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:5]


def _topic_words(text):
    return set(_topic_words_cached(str(text)))


@lru_cache(maxsize=20000)
def _topic_words_cached(text):  # recall re-reads the same memory texts on every query
    stop = _stopwords()
    return frozenset(_stem(w.strip("'’-")) for w in _WORD.findall(text.lower())
                     if len(w) >= 3 and w not in stop and w not in _GENERIC and not w.isdigit())


def _names(memory):
    """Stems of capitalised words that are not sentence-initial (a rough "names something" signal)."""
    text = str(memory.get("content", ""))
    out = set()
    for m in re.finditer(r"\b[A-Z][a-z]+\b", text):
        before = text[:m.start()].rstrip()
        if before and before[-1] not in ".!?:]":
            out.add(_stem(m.group(0).lower()))
    return out


def _enabled(name):
    return (settings.get(f"recall.{name}", {}) or {}).get("enabled", True)


def promote_updates(ranked, query="", window=5, consider_top=3, max_share=0.3, min_meaning_ratio=0.4):
    """
    ranked: candidates sorted by relevance (desc); query: the question. Reorders in place and returns the list.
    A candidate B within the top `window` is moved just above an earlier-stored candidate A
    among the top `consider_top` when B is later, carries a change cue and shares a topic word
    with A. B must also cover the question: it may not lack a name from the question that A matched
    ("Project Alpha deadline was moved" does not replace "Project Beta deadline" for a question about
    Beta), and if it lacks any other question word A matched, it must share at least two specific
    words with A ("grant report ... due" -> "grant report deadline moved": yes; "Kestrel team lead"
    -> "Kestrel deadline moved": no), and B's meaning match to the question must be at least
    40% of A's (B updates the part of A that was asked about). Each promoted memory gets "update_of": A's id (shown in its explanation).
    """
    if not _enabled("updates") or len(ranked) < 2 or asks_for_earlier(query):
        return ranked
    head = ranked[:window]
    # A shared word is a topic only if it is specific among the candidates: boilerplate that most
    # matched memories contain ("run", "turn", a product name in every message) links nothing.
    df = {}
    for m in ranked:
        for w in _topic_words(m.get("content", "")):
            df[w] = df.get(w, 0) + 1
    limit = max(3, max_share * len(ranked))
    topics = {id(m): {w for w in _topic_words(m.get("content", "")) if df.get(w, 0) <= limit} for m in head}
    asked = {w for w in _topic_words(query) if df.get(w, 0) <= limit}
    named = _named_in(query)
    for _ in range(window):  # repeat until stable: chains of updates (A -> B -> C)
        moved = False
        for i, a in enumerate(head[:consider_top]):
            for b in head[i + 1:]:
                if float(b.get("timestamp") or 0) <= float(a.get("timestamp") or 0):
                    continue
                if not _CHANGE_CUE.search(str(b.get("content", ""))):
                    continue
                shared = _same_subject(a, b, topics, asked, named, min_meaning_ratio)
                if not shared:
                    continue
                b["update_of"] = a.get("id")
                b["update_shared_words"] = sorted(shared)[:3]
                b["relevance"] = max(float(b.get("relevance") or 0.0), float(a.get("relevance") or 0.0) + 1e-6)
                head.remove(b)
                head.insert(i, b)
                moved = True
                break
            if moved:
                break
        if not moved:
            break
    # A later update that already ranks above what it updates needs no move, but it is still an update: mark it
    # (same tests as above), so the outdated memory's span is never shown as the answer (RecallEngine._best_answer).
    for b in head:
        if b.get("update_of") is not None or not _CHANGE_CUE.search(str(b.get("content", ""))):
            continue
        for a in head[head.index(b) + 1:consider_top]:
            if float(b.get("timestamp") or 0) <= float(a.get("timestamp") or 0):
                continue
            shared = _same_subject(a, b, topics, asked, named, min_meaning_ratio)
            if shared:
                b["update_of"] = a.get("id")
                b["update_shared_words"] = sorted(shared)[:3]
                break
    ranked[:window] = head
    return ranked

_EARLIER = re.compile(r"\b(?:originally|initially|at first|original|previous|previously|earlier|used to|"
                      r"before (?:switching|changing|the change|moving|replacing|returning|exchanging|it was))\b", re.I)


def asks_for_earlier(query):
    """"What ointment was ORIGINALLY applied?" asks for the fact an update replaced."""
    return bool(_EARLIER.search(str(query or "")))


_FIRST_PERSON = re.compile(r"\b(?:i|my|me|mine|i'm|i've)\b", re.I)
_PRESENT = re.compile(r"\b(?:is|are|am|do|does|'s|currently|now|these days|right now)\b", re.I)
_HISTORY = re.compile(r"\b(?:was|were|did|had|used to|ever|before|first|last time|previous(?:ly)?|originally|initially|"
                      r"earlier|history|in \d{4}|ago)\b", re.I)


def asks_about_now(query):
    """
    "Who is my manager?", "Where do I live?": a first-person question about the user's current state. When two
    memories give different answers to it (recall.answer_conflicts), the later one is the user's current fact.
    Third-person and history questions ("What is Caroline excited about?", "Who was my first manager?") are not:
    for those the reader's most confident answer stays (LoCoMo dev: later answers matched the gold far less
    often, eval/results/later_answer).
    """
    q = str(query or "")
    return bool(_FIRST_PERSON.search(q) and _PRESENT.search(q) and not _HISTORY.search(q) and not asks_for_earlier(q))


_FOR_WHAT = re.compile(r"\bfor\s+(.+?)\s*[?.!]*$", re.I)


def purpose_stems(query):
    """
    "Which credit card do I use for booking flights?" -> {"book", "flig"}: what the question narrows the topic to (the
    words after "for"), as 4-letter stems so "flight bookings" matches. Empty when there is no such phrase.
    """
    m = _FOR_WHAT.search(str(query or ""))
    if not m:
        return set()
    stop = _stopwords() | _FIRST_PERSON_WORDS
    return {w[:4] for w in _WORD.findall(m.group(1).lower()) if len(w) >= 3 and w not in stop}


def matches_purpose(content, stems):
    return bool(stems) and any(w[:4] in stems for w in _WORD.findall(str(content or "").lower()) if len(w) >= 3)


_FIRST_PERSON_WORDS = {"i", "my", "me", "mine", "we", "our", "us", "now", "currently", "today", "these", "days", "right"}
_CHANGED = re.compile(r"\b(?:shift(?:ed|ing)|switch(?:ed|ing)|transferr(?:ed|ing)|moved|changed|instead)\b", re.I)


def states_a_change(later, earlier, query):
    """
    "Which school does Meera go to?" with "meera goes to dps whitefield" and, later, "shifted meera to inventure
    academy this term": for a present question about someone else, the later answer replaces the earlier one only
    when its memory says something changed (shifted, switched, transferred, moved, changed, instead) and both
    memories name the same word of the question ("meera"). Without such a word the most confident answer stays
    (third person, see asks_about_now). eval/results/update_set
    """
    q = str(query or "")
    if _FIRST_PERSON.search(q) or not _PRESENT.search(q) or _HISTORY.search(q) or asks_for_earlier(q):
        return False
    later, earlier = str(later or ""), str(earlier or "")
    if not _CHANGED.search(later):
        return False
    stop = _stopwords()
    words = lambda t: {w for w in _WORD.findall(t.lower()) if len(w) >= 3 and w not in stop}
    return bool(words(q) & words(later) & words(earlier))


_ENDED = re.compile(r"\b(?:resigned|quit|left|closed|sold|cancel+ed|stopped|broke up|got rid of|gave (?:away|up)|"
                    r"no longer|not anymore|anymore|is dead|died|went to my|moved out of|ended|switched from)\b", re.I)


def ended_values(memories):
    """
    Words of things a memory says have ended for the user, with that memory's time: "resigned from infosys",
    "closed the hdfc account", "sold the honda", "cancelled cult", "the xps is dead", "the iphone went to my mom".
    [(timestamp, {words})]. Read per clause, so "bought a hyundai creta" next to "sold the honda" is not ended.
    """
    out = []
    for m in memories:
        for clause in _CLAUSE.split(str(m.get("content") or "")):
            if _ENDED.search(clause):
                words = {w for w in _WORD.findall(clause.lower()) if len(w) >= 3 and w not in _stopwords() and w not in _GENERIC}
                if words:
                    out.append((float(m.get("timestamp") or 0), words))
    return out


_FIRST_PERSON_TEXT = re.compile(r"\b(?:i|i'm|im|i've|i'd|my|me|mine|we|our|us)\b", re.I)
_VERB_FIRST = re.compile(r"^\W*(?:\w+ed|sold|bought|went|got|found|joined|took|left|made|quit|began|became|broke|gave|"
                         r"spent|switched|shifted|started|signed|moved|spilled|accepted|closed|cancel+ed|sent|paid|sat)\b", re.I)


def about_the_user(content):
    """
    The memory states something about the user: "I/my/me/we" in it, or a texting-style sentence with the subject left
    out ("sold the honda, bought a hyundai creta", "got moved to the search team"). "Anita lives in Bangalore" is not.
    """
    c = str(content or "")
    return bool(_FIRST_PERSON_TEXT.search(c) or _VERB_FIRST.match(c))


def _named_in(query):
    first = re.match(r"\s*(\w+)", query)
    return {_stem(w.lower()) for w in re.findall(r"\b[A-Z][a-z]+\b", query) if not (first and w == first.group(1))}


def _same_subject(a, b, topics, asked, named, min_meaning_ratio):
    """
    Shared specific words if B can be about the same thing as A for this question, else an empty set.
    B must share a topic word with A and cover the question: it may not lack a name from the
    question that A matched while naming something else ("Project Alpha deadline was moved" vs a
    question about Beta); if it lacks any other question word A matched it must share at least two
    specific words with A ("Kestrel team lead" vs "Kestrel deadline moved": no); and its meaning
    match to the question must be at least `min_meaning_ratio` of A's (B updates the asked part of A).
    """
    shared = topics[id(a)] & topics[id(b)]
    if not shared:
        return set()
    missing = (asked & topics[id(a)]) - topics[id(b)]
    if missing & named and _names(b) - _names(a) - named:
        return set()
    if missing - named and len(shared) < 2:
        return set()
    sa, sb = a.get("semantic_score"), b.get("semantic_score")
    if sa is not None and sb is not None and sb < min_meaning_ratio * sa:
        return set()  # A: "cover her night duty ... taking mom to the eye clinic"; B: "night shift on 28th instead"
    return shared


def _norm_answer(text):
    return " ".join(_WORD.findall(str(text or "").lower()))


def resolve_answer_conflicts(served, query, min_meaning_ratio=0.4):
    """
    Updates without a cue word (recall.answer_conflicts). After the answer checker has read the
    served memories, two of them that both answer the question with DIFFERENT answers and are about
    the same thing (same guards as promote_updates) disagree; the later one is served first, or
    the earlier one when the question asks for the earlier fact ("originally", "previously").
    "found a chambre de bonne on rue de la roquette" (later) over "my studio on boulevard
    saint-germain" needs no "moved to"; the reader already found both answers. Returns the list.
    """
    if not _enabled("answer_conflicts") or len(served) < 2:
        return served
    answered = [m for m in served if m.get("_answer_ok") and _norm_answer(m.get("answer"))]
    if len(answered) < 2:
        return served
    topics = {id(m): _topic_words(m.get("content", "")) for m in served}
    asked, named = _topic_words(query), _named_in(query)
    top = answered[0]
    group = [top]
    for b in answered[1:]:
        x, y = _norm_answer(top.get("answer")), _norm_answer(b.get("answer"))
        if x in y or y in x:
            continue  # the same answer (or one contains the other): agreement, not a conflict
        first, second = (top, b) if float(top.get("timestamp") or 0) <= float(b.get("timestamp") or 0) else (b, top)
        if _norm_answer(second.get("answer")) in _norm_answer(first.get("content")):
            continue  # the later answer is already in the earlier memory ("ferritin" in the prescription
            # message): it does not replace it ("ferrex 150"), a replacement names something new
        if _same_subject(first, second, topics, asked, named, min_meaning_ratio):
            group.append(b)
    if len(group) < 2:
        return served
    pick = (min if asks_for_earlier(query) else max)(group, key=lambda m: float(m.get("timestamp") or 0))
    if pick is not served[0]:
        pick["answer_conflict_with"] = [m.get("id") for m in group if m is not pick]
        served = [pick] + [m for m in served if m is not pick]
    return served


_PRONOUN = re.compile(r"\b(?:she|he|they|her|his|him|their|them|hers|theirs|the user['’]s|the user)\b", re.I)
_VERB_FIX = [  # agreement after the swap, so lexical search sees natural first-person text
    (r"\b(does|did) (?:she|he|they)\b", lambda m: ("do" if m.group(1).lower() == "does" else "did") + " I"),
    (r"\b(is|was) (?:she|he)\b", lambda m: ("am" if m.group(1).lower() == "is" else "was") + " I"),
    (r"\b(are|were) they\b", lambda m: ("am" if m.group(1).lower() == "are" else "was") + " I"),
    (r"\b(has|have) (?:she|he|they)\b", lambda m: "have I"),
    (r"\bis the user\b(?!['’]s)", lambda m: "am I"),
    (r"\bdoes the user\b(?!['’]s)", lambda m: "do I"),
]
_SWAP = {"she": "I", "he": "I", "they": "I", "her": "my", "his": "my", "their": "my", "him": "me", "them": "me",
         "hers": "mine", "theirs": "mine", "the user's": "my", "the user’s": "my", "the user": "I"}
_TITLE = re.compile(r"\b[A-Z][a-z]+\b")


def user_perspective(query, not_people=(), is_speaker=None):
    """'Which hospital does she work at?' -> 'Which hospital do I work at?'; unchanged when the
    question names a person before the first pronoun, names anyone recorded as a speaker in this
    user's memories (`is_speaker(name)`: "his travel dreams for Tim" in a Tim/John conversation),
    or has no third-person pronoun."""
    if not _enabled("user_perspective"):
        return query
    m = _PRONOUN.search(query)
    if not m:
        return query
    if is_speaker and any(is_speaker(w) for w in _TITLE.findall(query) if w not in not_people):
        return query
    before = query[:m.start()]
    words = _TITLE.findall(before)
    first = re.match(r"\s*([A-Z][a-z]+)", query)
    names = [w for w in words if w not in not_people and not (first and w == first.group(1) and before.lstrip().startswith(w))]
    if names:
        return query  # "When did Meera say she ..." - the pronoun is Meera
    out = query
    for pat, rep in _VERB_FIX:
        out = re.sub(pat, rep, out, flags=re.I)
    out = re.sub(r"\b(?:the user['’]s|the user|she|he|they|her|his|him|their|them|hers|theirs)\b",
                 lambda mm: _SWAP[mm.group(0).lower()], out, flags=re.I)
    if query[:1].isupper() and out[:1].islower():
        out = out[0].upper() + out[1:]
    return out


# --- who is the clause about ------------------------------------------------------------------
_RELATIONS = {"roommate", "flatmate", "housemate", "sister", "brother", "sis", "bro", "mom", "mum", "mother",
              "dad", "father", "wife", "husband", "hubby", "son", "daughter", "kid", "kids", "child", "children",
              "friend", "bestie", "coworker", "co-worker", "colleague", "boss", "neighbor", "neighbour", "cousin",
              "uncle", "aunt", "auntie", "partner", "girlfriend", "boyfriend", "fiance", "fiancee", "grandson",
              "granddaughter", "grandma", "grandpa", "nephew", "niece", "in-law", "classmate", "teammate"}
_SAME = {"mom": "mother", "mum": "mother", "dad": "father", "hubby": "husband", "sis": "sister", "bro": "brother",
         "kid": "child", "kids": "child", "children": "child", "son": "child", "daughter": "child",
         "co-worker": "coworker", "colleague": "coworker", "neighbour": "neighbor", "auntie": "aunt"}
_CLAUSE = re.compile(r"[,;.!?]+\s*|\s+(?:but|and|so|because|while|whereas)\s+", re.I)
_LEAD = re.compile(r"^(?:(?:btw|uff|ok|so|actually|also|then|our|my|his|her|the|a|an)\s+)*", re.I)
_PERSON_PRONOUN = re.compile(r"^(?:he|she|they|his|her|their)\b", re.I)
# "rahul keeps telling me golds gym is better": someone else (named or not) telling, advising or recommending something
# to the user. What they say is theirs, not a fact about the user.
_TELLS_ME = re.compile(r"^([a-z][a-z'’-]*)\s+(?:(?:keeps|kept|always|also|just|even)\s+)?(?:tell(?:s|ing)?|told|say(?:s|ing)?|said|"
                       r"ask(?:s|ed|ing)?|recommend(?:s|ed|ing)?|suggest(?:s|ed|ing)?|advis(?:es|ed|ing)|insist(?:s|ed|ing)?|"
                       r"swear(?:s|ing)?|swore|convinc(?:es|ed|ing)|push(?:es|ed|ing)|want(?:s|ed))\b.*\b(?:me|us)\b", re.I)
_ADVICE_TO_ME = re.compile(r"\b(?:me|us)\s+(?:to|not\s+to)\s+[a-z]", re.I)
_NOT_A_PERSON = {"i", "it", "this", "that", "there", "what", "which", "who", "we", "you", "everything", "nothing", "work",
                 "life", "the", "a", "an", "someone", "somebody"}


def _rel(w):
    w = w.lower().rstrip("'’s") if w.lower().endswith(("'s", "’s")) else w.lower()
    return _SAME.get(w, w)


def without_other_people(context, question):
    """
    The memory as the answer checker should read it for a question about the user: clauses whose
    subject is another person ("roommate liam failed his organic chemistry midterm", "my sister
    chloe is moving to austin", and a following "he/she ..." clause) are removed, unless the
    question asks about that person (names them or the relation: "Where is her husband
    traveling?"). The user's own clauses stay ("taking mom to the clinic": mom is not the subject).
    A memory entirely about someone else becomes empty: it cannot answer a question about the user.
    Found on unseen Gemini batch 3: "What major is the athlete studying in college?" was answered
    "organic chemistry" from the roommate's midterm. Config recall.answer_check.other_people.
    """
    cfg = (settings.get("recall.answer_check", {}) or {}).get("other_people") or {}
    if not cfg.get("enabled", True) or not context:
        return context
    q_words = {_rel(w) for w in _WORD.findall(str(question).lower())}
    parts, last, kept, dropped, prev_other = [], 0, [], False, False
    for m in _CLAUSE.finditer(context):
        parts.append((context[last:m.start()], context[m.start():m.end()]))
        last = m.end()
    parts.append((context[last:], ""))
    for clause, sep in parts:
        body = _LEAD.sub("", clause.strip())
        words = _WORD.findall(body.lower())[:3]
        rel = words[0] if words and (_rel(words[0]) in _RELATIONS or words[0] in _RELATIONS) else None  # subject position
        other = False
        if rel is not None:
            # "my sister chloe": the word after the relation is a name; "my sister's manager": a possessive relation
            # is followed by what that person has (their manager, not a name), which a question about the user's
            # manager must not be taken to ask about
            possessive = rel.endswith(("'s", "’s"))
            name = words[words.index(rel) + 1] if not possessive and words.index(rel) + 1 < len(words) else None
            asked_about = _rel(rel) in q_words or (name and name in q_words)
            other = not asked_about
        elif prev_other and _PERSON_PRONOUN.match(body):
            other = True  # "he is locking himself in the library" continues the roommate's clause
        elif ((t := _TELLS_ME.match(body.lower())) and t.group(1) not in _NOT_A_PERSON
              and not _ADVICE_TO_ME.search(body)):  # "the doctor told me to take vitamin d" is about the user
            other = t.group(1) not in q_words  # "rahul keeps telling me ..." unless the question is about rahul
            rel = rel if not other else t.group(1)  # so a following "he goes there" continues it
        prev_other = other if (rel is not None or other) else False
        if other:
            dropped = True
            continue
        kept.append(clause + sep)
    if not dropped:
        return context
    return re.sub(r"(?:\s+(?:but|and|so|because|while|whereas))?[\s,;]*$", "", "".join(kept))
