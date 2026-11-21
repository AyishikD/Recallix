"""
Rule-based (entity, relation, value) fact extraction.

Replaces the LLM triad-extraction call on the store path. Patterns run per
sentence, most specific first, and each value span can be claimed only once.

Forms covered (entity "user" = the speaker):
    my <rel> is <value>                      user::<rel>
    <value> is my <rel>                      user::<rel>
    I have / we have a <rel> named <value>   user::<rel>
    <Entity>'s <rel> is <value>              <entity>::<rel>
    the <rel> of <entity> is <value>         <entity>::<rel>
    [the] <Entity> <known-rel> is <value>    <entity>::<rel>    "Project Alpha deadline is ..."
    our <rel> is <value>                     user::<rel>
    the <event> is on/at <time>              <event>::when       "The standup is at 10am"
    <X> is also known as / called <Y>        <y>::alias_of = X
    I live in / work at / am a / am N years old / am allergic to / call me ...
    <Name> lives in / works at ...

"is" and state-transition verbs ("moved to", "pushed to", "postponed to")
share a slot, so the TruthEngine keeps the newest value.

Precision guard: a bare "my X is Y" is accepted only if X is a known
attribute noun or Y looks like a concrete value (capitalised word, digit,
currency, email, quoted). "my cat is sleeping" therefore yields nothing,
while "my cat is named Pixel" and "my budget is $500" do.
"""
import re
import time

_TRANSITION = r"(?:changed|moved|updated|switched|shifted|set|pushed|postponed|delayed|bumped)"
_VERB = (  # longer, state-transition forms first so "was moved to X" yields the value X
    rf"(?:ha(?:s|ve)\s+(?:been\s+)?{_TRANSITION}\s+(?:to|for)"
    rf"|(?:got|was|were|is)\s+{_TRANSITION}\s+(?:to|for)"
    rf"|{_TRANSITION}\s+(?:to|for)"
    r"|is\s+now|is\s+named|is\s+called|is|are|was|were|will\s+be|=)"
)
_VALUE = r"(?P<val>[^;!?\n]{1,80})"  # sentences are pre-split, so dots (emails, Dr.) are allowed
_REL = r"(?P<rel>[a-z][a-z0-9' -]{0,30}?)"
_ENTITY_CAP = r"(?P<ent>(?:[A-Z][\w-]*\s+){0,3}[A-Z][\w-]*)"

# Attribute nouns that may directly follow a named entity, and that make a
# "my <rel> is <value>" statement trustworthy even with a lowercase value.
_KNOWN_RELS = (
    r"(?:launch|start|end|due|delivery|go-live|cutoff)\s+date|deadline|date|time|"
    r"name|full\s+name|nickname|status|state|phase|stage|code|password|passcode|pin|address|"
    r"phone(?:\s+number)?|number|email|birthday|location|city|country|hometown|budget|owner|"
    r"lead|manager|version|price|cost|salary|goal|role|title|"
    r"job|age|priority|capital|timezone|time\s+zone|"
    r"favou?rite\s+[a-z]+"
)
_KNOWN_REL_RE = re.compile(rf"^(?:{_KNOWN_RELS})$", re.I)

_PATTERNS = [
    # alias: "<X> is also known as / aka <Y>"
    ("alias", re.compile(
        r"\b(?:the\s+)?(?P<val>[\w -]{2,40}?)\s+(?:is\s+)?(?:also\s+)?(?:\w+ly\s+)?(?:known\s+as|aka|nicknamed)\s+"
        r"['\"‘“]?(?P<ent>[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,3})", re.I), None),
    # I have a <rel> named <value>
    ("named", re.compile(rf"\bI\s+(?:have|own|got)\s+(?:a|an)\s+{_REL}\s+(?:named|called)\s+{_VALUE}", re.I), "user"),
    # <value> is my <rel>
    ("inverse", re.compile(rf"(?:^|\b)(?P<val>[A-Z][\w-]*(?:\s+[A-Z][\w-]*){{0,2}})\s+is\s+(?:my|our)\s+{_REL}\s*$"), "user"),
    # my / our <rel> is <value>
    ("my", re.compile(rf"\b(?:my|our)\s+{_REL}\s+{_VERB}\s+{_VALUE}", re.I), "user"),
    # the user's <rel> is <value>
    ("my", re.compile(rf"\bthe\s+user'?s\s+{_REL}\s+{_VERB}\s+{_VALUE}", re.I), "user"),
    # <Entity>'s <rel> is <value>
    ("possessive", re.compile(rf"{_ENTITY_CAP}'s\s+{_REL}\s+(?i:{_VERB})\s+{_VALUE}"), None),
    # the <rel> of <entity> is <value>
    ("of", re.compile(rf"\bthe\s+{_REL}\s+of\s+(?:the\s+)?(?P<ent>[\w-]+(?:\s+[\w-]+){{0,3}}?)\s+{_VERB}\s+{_VALUE}", re.I), None),
    # [the] <Entity> <known-rel> is <value>   (entity Capitalised; rel/verb any case)
    ("entity_rel", re.compile(rf"(?:\b[Tt]he\s+)?{_ENTITY_CAP}\s+(?P<rel>(?i:{_KNOWN_RELS}))\s+(?i:{_VERB})\s+{_VALUE}"), None),
    # <Name-with-code> <Attribute>: <value>   ("Patient X-100 Status: critical")
    ("code_colon", re.compile(
        rf"(?P<ent>(?:[A-Z][\w-]*\s+){{0,3}}[A-Za-z]*[\w-]*\d[\w-]*)\s+(?P<rel>[A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){{0,2}})\s*:\s*{_VALUE}"), None),
    # [the] <Entity> <noun phrase> is <Value>   (open relation; guarded: value must look concrete)
    ("entity_np", re.compile(rf"(?:\b[Tt]he\s+)?{_ENTITY_CAP}\s+(?P<rel>[a-z][a-z-]*(?:\s+[a-z][a-z-]*){{0,2}})\s+(?i:{_VERB})\s+{_VALUE}"), None),
    # the <event> is on/at <time>   /   the <event> was moved to <time>
    ("event", re.compile(
        rf"\b(?:the\s+)?(?P<ent>[a-z][a-z-]*(?:\s+[a-z][a-z-]*)??)\s+(?:{_VERB}|is\s+scheduled|runs)\s+"
        r"(?P<val>(?:on|at|for|from|by)\s+[^.;!?\n]{1,60}|(?:the\s+)?\d[^.;!?\n]{0,40}|"
        r"(?:next|this)\s+\w+|(?:mon|tues|wednes|thurs|fri|satur|sun)day\b[^.;!?\n]{0,30}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d[^.;!?\n]{0,30})", re.I), "__event__"),
]

_FIRST_PERSON = [
    (re.compile(rf"\b(?:I\s+live|I'm\s+living|I\s+am\s+living|I\s+moved)\s+(?:in|to)\s+{_VALUE}", re.I), "location"),
    (re.compile(rf"\bI\s+work\s+(?:at|for)\s+{_VALUE}", re.I), "employer"),
    (re.compile(rf"\b(?:I\s+am|I'm)\s+(?P<val>\d{{1,3}})\s+(?:years?\s+old|y/?o)\b", re.I), "age"),
    (re.compile(rf"\b(?:I\s+am|I'm)\s+allergic\s+to\s+{_VALUE}", re.I), "allergy"),
    (re.compile(rf"\b(?:I\s+am|I'm)\s+(?:an?)\s+{_VALUE}", re.I), "role"),
    (re.compile(rf"\b(?:call\s+me|I\s+go\s+by)\s+{_VALUE}", re.I), "name"),
    (re.compile(rf"\bI\s+was\s+born\s+(?:on|in)\s+{_VALUE}", re.I), "birth"),
]

_CARRY = re.compile(rf"^(?P<rel>[A-Za-z][a-z]+(?:\s+[a-z]+){{0,2}})\s+(?i:{_VERB})\s+{_VALUE}")

_THIRD_PERSON = [
    (re.compile(rf"\b{_ENTITY_CAP}\s+(?:lives|is\s+living|moved)\s+(?:in|to)\s+{_VALUE}"), "location"),
    (re.compile(rf"\b{_ENTITY_CAP}\s+works\s+(?:at|for)\s+{_VALUE}"), "employer"),
]

_REL_SYNONYMS = {
    "due_date": "deadline",
    "phone_number": "phone",
    "full_name": "name",
    "cats_name": "cat",
    "dogs_name": "dog",
    "time_zone": "timezone",
    "city": "location",
    "hometown": "location",
}

_STOP_ENTITIES = {"it", "this", "that", "there", "he", "she", "they", "we", "i", "you", "what", "which",
                  "my", "our", "your", "the", "a", "an", "hi", "hello", "hey", "who", "where", "when"}

# Event subjects worth tracking as "<event>::when". Generic nouns that are
# rarely calendar events are excluded to protect precision.
_EVENT_STOP = {"weather", "day", "week", "it", "this", "that", "flight", "food", "coffee", "movie", "build",
               "tests", "test", "sky", "traffic", "match", "game", "trip"}

_CLAUSE_BREAK = re.compile(r"\s*(?:,\s*)?\b(?:and|but|while|because|so|although)\s+(?:i|i'm|my|we|our|he|she|they)\b.*$", re.I)
_LEADING_FILLER = re.compile(r"^(?:still|now|actually|also|just|really|currently|officially|on)\s+", re.I)
_TRAILING_FILLER = re.compile(r",?\s*(?:by the way|btw|i think|i guess|for now|lol|it's serious)\s*$", re.I)
_CONCRETE = re.compile(r"[A-Z]|\d|[$€£₹@]|^['\"]")
_NOT_A_VALUE = re.compile(r"^(?:not|no|very|so|too|going|getting|being|really|quite|pretty|a\s+bit|kind\s+of|\w+ing)\b", re.I)


def _norm_token(s):
    return re.sub(r"[^a-z0-9_]+", "", s.lower())


def _norm_entity(ent):
    ent = re.sub(r"^(?:the|a|an)\s+", "", ent.strip(), flags=re.I)
    return _norm_token(re.sub(r"\s+", "_", ent.strip().lower())) if " " in ent.strip() and ent.strip().islower() \
        else _norm_token(ent)


def _norm_rel(rel):
    rel = re.sub(r"^(?:the|a|an)\s+", "", rel.strip().lower())
    rel = rel.replace("'s", "").replace("favourite", "favorite")
    rel = re.sub(r"\s+", "_", rel)
    rel = _norm_token(rel)
    return _REL_SYNONYMS.get(rel, rel)


def _clean_value(val):
    val = _CLAUSE_BREAK.sub("", val)
    val = _TRAILING_FILLER.sub("", val.strip())
    val = _LEADING_FILLER.sub("", val)
    return val.strip().strip("'\"").rstrip(",:").strip()


def _looks_like_fact(rel, val):
    """Precision guard for open-ended 'X is Y' statements."""
    if _KNOWN_REL_RE.match(rel.strip()):
        return not _NOT_A_VALUE.match(val)
    return bool(_CONCRETE.search(val)) and not _NOT_A_VALUE.match(val)


def extract_triads(message):
    """Return a list of {"entity", "relation", "value"} dicts (raw, un-normalised)."""
    from backend.brain.features.query import capitalised_spans
    triads = []
    seen = set()
    last_entity = None  # most recent named entity in this message (for subject-less follow-ups)

    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", message):
        sentence = sentence.strip().rstrip(".!?;")
        if not sentence:
            continue
        named_here = capitalised_spans(sentence, skip_sentence_initial_words=True)
        claimed = []  # value spans already explained by a more specific pattern

        def overlaps(m):
            s, e = m.start("val"), m.end("val")
            return any(s < ce and cs < e for cs, ce in claimed)

        def add(m, ent, rel, val, guard=True):
            if not ent or not rel or not val or ent.lower() in _STOP_ENTITIES:
                return
            if guard and not _looks_like_fact(rel, val):
                return
            key = (ent.lower(), rel.lower())
            if key in seen:
                return
            seen.add(key)
            claimed.append((m.start("val"), m.end("val")))
            triads.append({"entity": ent, "relation": rel, "value": val})

        for kind, pattern, fixed_entity in _PATTERNS:
            for m in pattern.finditer(sentence):
                if overlaps(m):
                    continue
                val = _clean_value(m.group("val"))
                if kind == "alias":
                    add(m, m.group("ent").strip(), "alias_of", val, guard=False)
                elif kind == "event":
                    ent = m.group("ent").strip()
                    if ent.lower() in _EVENT_STOP or ent.split()[-1].lower() in _EVENT_STOP:
                        continue
                    add(m, ent, "when", val, guard=False)
                elif kind in ("named", "inverse"):
                    add(m, fixed_entity, m.group("rel"), val, guard=False)
                else:
                    ent = fixed_entity or m.group("ent").strip()
                    if kind == "entity_np" and m.start("ent") == 0 and " " not in ent and not re.search(r"\d", ent):
                        continue  # a lone capitalised first word is not evidence of a name
                    add(m, ent, m.group("rel"), val, guard=(kind in ("my", "possessive", "of", "entity_np", "code_colon")))
        for pattern, rel in _FIRST_PERSON:
            m = pattern.search(sentence)
            if m and not overlaps(m):
                add(m, "user", rel, _clean_value(m.group("val")), guard=False)
        for pattern, rel in _THIRD_PERSON:
            m = pattern.search(sentence)
            if m and not overlaps(m):
                add(m, m.group("ent").strip(), rel, _clean_value(m.group("val")), guard=False)
        if last_entity and not claimed:  # no fact of its own: "<Relation> is now <Value>" about the last subject
            m = _CARRY.match(sentence)
            if m:
                add(m, last_entity, m.group("rel"), _clean_value(m.group("val")), guard=True)
        if named_here and not _CARRY.match(sentence):
            last_entity = "".join(named_here[0])  # a sentence's first named entity is usually its subject
    return triads


def to_slots(triads):
    """Normalise triads into the metadata slot format consumed by the TruthEngine."""
    slots = []
    version = int(time.time() * 1000)
    for t in triads:
        e_raw, r_raw, v = t.get("entity", ""), t.get("relation", ""), t.get("value", "")
        if not e_raw or not r_raw:
            continue
        e_norm = _norm_entity(e_raw)
        r_norm = _norm_rel(r_raw)
        if not e_norm or not r_norm:
            continue
        slots.append({
            "slot_id": f"{e_norm}::{r_norm}",
            "entity": e_norm,
            "relation": r_norm,
            "value": v,
            "version": version,
            "truth_status": "active",
        })
    return slots
