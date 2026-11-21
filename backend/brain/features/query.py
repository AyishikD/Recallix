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

_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i've|i'd|i'll)\b", re.I)
_CAP_SPAN = re.compile(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*")
_BRACKET_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")   # e.g. a "[date]" header before the text
_QUESTION_WORDS = {"what", "when", "where", "who", "whom", "which", "why", "how", "do", "does", "did",
                   "is", "are", "was", "were", "can", "could", "should", "would", "will", "i", "the",
                   "a", "an", "tell", "please", "any", "is there", "what's", "whats"}


def norm_id(s):
    s = re.sub(r"['’]s\b", "", s.strip())
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _norm_token(t):
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"['’]s$", "", t.lower()))


def _strip_leading_question_words(span):
    words = span.split()
    while words and words[0].lower().strip("'") in _QUESTION_WORDS:
        words.pop(0)
    return " ".join(words)


def capitalised_spans(text, skip_sentence_initial_words=False, skip_vocatives=False):
    """
    Capitalised spans as tuples of normalised tokens ("Galaxy X-99 Primary" ->
    ("galaxy", "x99", "primary")), leading question words/articles removed.

    skip_sentence_initial_words: drop a single-word span at the start of a
      sentence unless it contains a digit ("Status of X-77?" -> only ("x77",)).
      Used for questions, where a capitalised first word is usually not a name.
    skip_vocatives: drop direct address — a name immediately followed by , ! or ?
      ("Thanks, Caroline!" does not make the text about Caroline).
    """
    out = []
    for m in _CAP_SPAN.finditer(text):
        span = _strip_leading_question_words(m.group(0))
        if not span or span.lower() in _QUESTION_WORDS:
            continue
        tokens = tuple(t for t in (_norm_token(w) for w in span.split()) if t)
        if not tokens or len("".join(tokens)) < 2:
            continue
        before = text[:m.start()].rstrip()
        sentence_initial = not before or before[-1] in ".!?:\n"
        if skip_sentence_initial_words and sentence_initial and m.group(0) == span and len(tokens) == 1 \
                and not any(c.isdigit() for c in tokens[0]):
            continue
        if skip_vocatives and re.match(r"\s*[,!?]", text[m.end():]):
            continue
        out.append(tokens)
    return out


def capitalised_entities(text):
    """Normalised ids of capitalised spans, ignoring leading question/aux words."""
    return ["".join(t) for t in capitalised_spans(text)]


def spans_match(a, b):
    """
    Token spans name the same entity if equal, if one is a contiguous run of the
    other ("x99" in "galaxy x99 primary emission"; "kestrel" in "project kestrel"),
    or if their joined forms are a one-edit typo of each other.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    n = len(short)
    if any(long_[i:i + n] == short for i in range(len(long_) - n + 1)):
        return True
    return entity_matches("".join(a), "".join(b))


def strip_header(text):
    """Text without a leading bracketed header (e.g. a timestamp)."""
    return _BRACKET_PREFIX.sub("", text, count=1)


def is_first_person(text):
    return bool(_FIRST_PERSON.search(text))


def expand_relations(words):
    words = {w.lower() for w in words}
    rels = set(words)
    for group in RELATION_GROUPS:
        if words & group:
            rels |= group
    return rels


def analyze_query(query):
    spans = capitalised_spans(query, skip_sentence_initial_words=True)
    strong = ["".join(s) for s in spans]
    if _FIRST_PERSON.search(query):
        strong.append("user")
        spans.append(("user",))
    words = re.findall(r"[a-z][a-z'-]*", query.lower())
    weak = [norm_id(t) for t in content_terms(query)]
    return {
        "strong_spans": list(dict.fromkeys(spans)),
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
    # Typo tolerance is for spelling: identifiers with digits ("x101" vs "x102") must match exactly.
    if any(c.isdigit() for c in a + b):
        return False
    return len(short) >= 5 and _edit_distance_at_most_one(a, b)


def relation_matches(slot_relation, query_relations):
    if not query_relations:
        return False
    parts = set(slot_relation.split("_")) | {slot_relation}
    return bool(expand_relations(parts) & query_relations)


def slot_matches(slot, analysis, allow_weak=True):
    """Named (strong) entities match fuzzily; plain words (weak) only match a slot entity exactly."""
    ent = slot.get("entity", "")
    ent_ok = any(entity_matches(ent, e) for e in analysis["strong_entities"]) or \
        (allow_weak and ent in analysis["weak_entities"])
    return ent_ok and relation_matches(slot.get("relation", ""), analysis["relations"])


def memory_entities(memory_text, slots):
    """Entity ids a memory talks about: its slot entities, capitalised spans, and 'user'."""
    ents = {s.get("entity", "") for s in slots}
    ents.update(capitalised_entities(memory_text))
    if _FIRST_PERSON.search(memory_text):
        ents.add("user")
    return {e for e in ents if e}


def memory_subject(memory_text, speaker=None):
    """
    Who a memory is about (answer-level attribution):
      owner     the speaker of a first-person statement ("I ran a race" said by
                Melanie is about Melanie); the user if no speaker is recorded
      mentions  named entities in the text, excluding direct address
                ("Thanks, Caroline!") and the header
    `speaker` comes from the store call's metadata, never guessed from the text.
    """
    body = strip_header(memory_text)
    if speaker:
        sp = re.escape(speaker.strip())
        body = re.sub(rf"^\s*{sp}\s*:\s*", "", body, count=1)
    owner = None
    if is_first_person(body):
        owner = tuple(t for t in (_norm_token(w) for w in speaker.split()) if t) if speaker else ("user",)
    mentions = capitalised_spans(body, skip_vocatives=True)
    return owner, mentions


# First-person forms and their third-person rewrites ({s} = speaker). English grammar only.
_FIRST_PERSON_REWRITES = [
    (r"\bI'm\b|\bI am\b", "{s} is"), (r"\bI've\b|\bI have\b", "{s} has"), (r"\bI'd\b", "{s} would"),
    (r"\bI'll\b", "{s} will"), (r"\bI was\b", "{s} was"), (r"\bmyself\b", "{s}"), (r"\bmine\b", "{s}'s"),
    (r"\bmy\b", "{s}'s"), (r"\bme\b", "{s}"), (r"\bI\b", "{s}"),
]


def resolve_first_person(text, speaker):
    """
    Rewrites a speaker's first-person statement in the third person, so a reader that
    does not know who "I" is can tell whose statement it is:
      ("Melanie", "Melanie: I'm starting to realize...") -> "Melanie: Melanie is starting to realize..."
    Text without a known speaker is returned unchanged.
    """
    if not speaker:
        return text
    head = re.match(r"^\s*(?:\[[^\]]*\]\s*)?(?:" + re.escape(speaker.strip()) + r"\s*:\s*)?", text)
    prefix, body = text[:head.end()], text[head.end():]
    for pattern, repl in _FIRST_PERSON_REWRITES:
        body = re.sub(pattern, repl.format(s=speaker.strip()), body, flags=re.I if pattern != r"\bI\b" else 0)
    return prefix + body
