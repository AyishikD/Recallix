"""
Query analysis and slot matching for recall (no LLM, no domain-specific rules).

analyze_query(q) returns:
  strong_entities  named things the question is about: capitalised spans
                   ("Project Kestrel", "Anita") and "user" for first-person
                   questions ("my", "I", "me").
  weak_entities    content words; these let "What time is standup?" match an
                   event slot "standup::when".
  relations        relation words the question asks about, expanded through
                   English synonym groups ("where ... live" -> location,
                   "due" -> deadline, "how much money" -> budget).

Matching is fuzzy where it should be: entity ids match on equality, on
containment ("kestrel" in "projectkestrel"), or on a string-similarity ratio
>= 0.85 (typos such as "Kestral").
"""
import re

from backend.brain.memory_ranking.importance_ranker import content_terms

# Each group is a set of words that ask about / name the same relation.
RELATION_GROUPS = [
    {"location", "live", "lives", "living", "based", "city", "reside", "home", "where", "moved", "hometown"},
    {"employer", "work", "works", "working", "job", "company", "employ", "employs", "employed"},
    {"name", "called", "named", "call", "alias", "alias_of"},
    {"birthday", "birth", "born"},
    {"age", "old"},
    {"deadline", "due", "deliver", "delivery"},
    {"when", "date", "time", "schedule", "scheduled", "day"},
    {"budget", "money", "cost", "price", "spend", "funding", "funds", "much"},
    {"allergy", "allergic", "allergies", "eat", "intolerant"},
    {"phone", "number", "mobile"},
    {"lead", "leads", "manager", "manages", "owner", "owns", "head", "runs"},
    {"email", "mail", "address"},
    {"role", "profession", "occupation"},
    {"status", "state", "phase", "stage"},
    {"password", "code", "passcode", "pin"},
]

_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i've|i'd)\b", re.I)
_CAP_SPAN = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*")
_QUESTION_WORDS = {"what", "when", "where", "who", "whom", "which", "why", "how", "do", "does", "did",
                   "is", "are", "was", "were", "can", "could", "should", "would", "will", "i", "the",
                   "a", "an", "tell", "please", "any", "is there", "what's", "whats"}


def norm_id(s):
    s = re.sub(r"'s\b", "", s.strip())
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _strip_leading_question_words(span):
    words = span.split()
    while words and words[0].lower().strip("'") in _QUESTION_WORDS:
        words.pop(0)
    return " ".join(words)


def capitalised_entities(text):
    """Normalised ids of capitalised spans, ignoring leading question/aux words."""
    out = []
    for span in _CAP_SPAN.findall(text):
        span = _strip_leading_question_words(span)
        if span and span.lower() not in _QUESTION_WORDS:
            nid = norm_id(span)
            if len(nid) > 1:
                out.append(nid)
    return out


def expand_relations(words):
    words = {w.lower() for w in words}
    rels = set(words)
    for group in RELATION_GROUPS:
        if words & group:
            rels |= group
    return rels


def analyze_query(query):
    strong = capitalised_entities(query)
    if _FIRST_PERSON.search(query):
        strong.append("user")
    words = re.findall(r"[a-z][a-z'-]*", query.lower())
    weak = [norm_id(t) for t in content_terms(query)]
    return {
        "strong_entities": list(dict.fromkeys(strong)),
        "weak_entities": list(dict.fromkeys(weak)),
        "relations": expand_relations(words),
        "asks_relation": any(set(words) & group for group in RELATION_GROUPS),
    }


def _edit_distance_at_most_one(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            if len(a) == len(b):
                i += 1
            j += 1
        else:
            i += 1
            j += 1
    return edits + (len(b) - j) <= 1


def entity_matches(a, b):
    """
    Same entity if the ids are equal; or one is the other plus a separate word
    (a prefix/suffix remainder of >= 3 chars: "projectkestrel" ~ "kestrel");
    or they differ by a single edit and are long enough for that to be a typo
    ("kestral" ~ "kestrel"). "falconet" is NOT "falcon".
    """
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and (long_.startswith(short) or long_.endswith(short)) and len(long_) - len(short) >= 3:
        return True
    return len(short) >= 5 and _edit_distance_at_most_one(a, b)


def relation_matches(slot_relation, query_relations):
    if not query_relations:
        return False
    parts = set(slot_relation.split("_")) | {slot_relation}
    return bool(expand_relations(parts) & query_relations)


def slot_matches(slot, analysis, allow_weak=True):
    ents = analysis["strong_entities"] + (analysis["weak_entities"] if allow_weak else [])
    ent_ok = any(entity_matches(slot.get("entity", ""), e) for e in ents)
    return ent_ok and relation_matches(slot.get("relation", ""), analysis["relations"])


def memory_entities(memory_text, slots):
    """Entity ids a memory talks about: its slot entities, capitalised spans, and 'user'."""
    ents = {s.get("entity", "") for s in slots}
    ents.update(capitalised_entities(memory_text))
    if _FIRST_PERSON.search(memory_text):
        ents.add("user")
    return {e for e in ents if e}
