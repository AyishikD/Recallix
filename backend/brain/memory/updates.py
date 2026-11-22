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
    stop = _stopwords()
    return {_stem(w.strip("'’-")) for w in _WORD.findall(str(text).lower())
            if len(w) >= 3 and w not in stop and w not in _GENERIC and not w.isdigit()}


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
    if not _enabled("updates") or len(ranked) < 2:
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
    first = re.match(r"\s*(\w+)", query)
    named = {_stem(w.lower()) for w in re.findall(r"\b[A-Z][a-z]+\b", query) if not (first and w == first.group(1))}
    for _ in range(window):  # repeat until stable: chains of updates (A -> B -> C)
        moved = False
        for i, a in enumerate(head[:consider_top]):
            for b in head[i + 1:]:
                if float(b.get("timestamp") or 0) <= float(a.get("timestamp") or 0):
                    continue
                if not _CHANGE_CUE.search(str(b.get("content", ""))):
                    continue
                shared = topics[id(a)] & topics[id(b)]
                if not shared:
                    continue
                missing = (asked & topics[id(a)]) - topics[id(b)]
                if missing & named and _names(b) - _names(a) - named:
                    continue  # A matches a name in the question that B lacks, and B names something else:
                    # "The Project Alpha deadline was moved" is not an update of "Project Beta deadline".
                    # Leaving the subject out is fine: "update: ballet got rescheduled" is still about Meera.
                if missing - named and len(shared) < 2:
                    continue  # "Who leads Kestrel?": B ("Kestrel deadline moved") shares only the name
                sa, sb = a.get("semantic_score"), b.get("semantic_score")
                if sa is not None and sb is not None and sb < min_meaning_ratio * sa:
                    continue  # B updates another part of A, not what was asked (A: "cover her night duty
                    # ... taking mom to the eye clinic"; B: "agreed for night shift on 28th instead")
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
    ranked[:window] = head
    return ranked


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
