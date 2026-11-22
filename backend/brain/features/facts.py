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
import hashlib
import os
import re
import time

# Identifies the exact rule set that extracted a fact: a hash of this file, so any
# change to a pattern changes the version. Stored on every slot (provenance) and
# compared by replay to explain why re-processing history changed a fact.
with open(os.path.abspath(__file__), "rb") as _f:
    EXTRACTOR_VERSION = "rules-" + hashlib.sha256(_f.read()).hexdigest()[:12]

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


def _sentences(message):
    """(sentence, offset of its first character in `message`) — offsets make provenance spans exact."""
    pos = 0
    for raw in re.split(r"(?<=[.!?;])\s+|\n+", message):
        start = message.find(raw, pos) if raw else pos
        pos = start + len(raw)
        lead = len(raw) - len(raw.lstrip())
        yield raw.strip().rstrip(".!?;"), start + lead


def extract_triads(message):
    """
    Return a list of {"entity", "relation", "value", "rule", "span"} dicts (raw, un-normalised).
    `rule` names the pattern that fired; `span` is [start, end) of the value in `message`.
    """
    from backend.brain.features.query import capitalised_spans
    triads = []
    seen = set()
    last_entity = None  # most recent named entity in this message (for subject-less follow-ups)

    for sentence, offset in _sentences(message):
        if not sentence:
            continue
        named_here = capitalised_spans(sentence, skip_sentence_initial_words=True)
        claimed = []  # value spans already explained by a more specific pattern

        def overlaps(m):
            s, e = m.start("val"), m.end("val")
            return any(s < ce and cs < e for cs, ce in claimed)

        def add(m, ent, rel, val, guard=True, rule=""):
            if not ent or not rel or not val or ent.lower() in _STOP_ENTITIES:
                return
            if guard and not _looks_like_fact(rel, val):
                return
            key = (ent.lower(), rel.lower())
            if key in seen:
                return
            seen.add(key)
            # Claim (and cite) only the text kept as the value: the raw match can run on
            # ("Priya and I live in Pune"), and claiming all of it would hide the next fact.
            raw = m.group("val")
            i = raw.find(val)
            start, end = (m.start("val") + i, m.start("val") + i + len(val)) if i >= 0 else (m.start("val"), m.end("val"))
            claimed.append((start, end))
            triads.append({"entity": ent, "relation": rel, "value": val, "rule": rule,
                           "span": [offset + start, offset + end]})

        for kind, pattern, fixed_entity in _PATTERNS:
            for m in pattern.finditer(sentence):
                if overlaps(m):
                    continue
                val = _clean_value(m.group("val"))
                if kind == "alias":
                    add(m, m.group("ent").strip(), "alias_of", val, guard=False, rule=kind)
                elif kind == "event":
                    ent = m.group("ent").strip()
                    if ent.lower() in _EVENT_STOP or ent.split()[-1].lower() in _EVENT_STOP:
                        continue
                    add(m, ent, "when", val, guard=False, rule=kind)
                elif kind in ("named", "inverse"):
                    add(m, fixed_entity, m.group("rel"), val, guard=False, rule=kind)
                else:
                    ent = fixed_entity or m.group("ent").strip()
                    if kind == "entity_np" and m.start("ent") == 0 and " " not in ent and not re.search(r"\d", ent):
                        continue  # a lone capitalised first word is not evidence of a name
                    add(m, ent, m.group("rel"), val, guard=(kind in ("my", "possessive", "of", "entity_np", "code_colon")), rule=kind)
        for pattern, rel in _FIRST_PERSON:
            m = pattern.search(sentence)
            if m and not overlaps(m):
                add(m, "user", rel, _clean_value(m.group("val")), guard=False, rule="first_person")
        for pattern, rel in _THIRD_PERSON:
            m = pattern.search(sentence)
            if m and not overlaps(m):
                add(m, m.group("ent").strip(), rel, _clean_value(m.group("val")), guard=False, rule="third_person")
        if last_entity and not claimed:  # no fact of its own: "<Relation> is now <Value>" about the last subject
            m = _CARRY.match(sentence)
            if m:
                add(m, last_entity, m.group("rel"), _clean_value(m.group("val")), guard=True, rule="carry")
        if named_here and not _CARRY.match(sentence):
            last_entity = "".join(named_here[0])  # a sentence's first named entity is usually its subject
    return triads


def to_slots(triads, timestamp=None):
    """
    Normalise triads into the metadata slot format consumed by the TruthEngine.
    `version` derives from the memory's timestamp (not the wall clock), so replaying a
    message with its original timestamp reproduces the slot exactly.
    """
    slots = []
    version = int((time.time() if timestamp is None else timestamp) * 1000)
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
            "source": {"rule": t.get("rule", ""), "span": t.get("span"), "extractor": EXTRACTOR_VERSION},
        })
    return slots


def resolve_definite(slots, lookup):
    """
    Coreference across messages for definite descriptions: in "The deadline was moved to
    December 18", "the deadline" refers to a deadline mentioned earlier ("Project Alpha
    deadline is December 12"). The event pattern alone would file the new value under an
    entity called "deadline"; this rewrites it to the earlier fact's entity so the
    TruthEngine sees one fact that changed.

    `lookup(relation)` returns the entity of this user's most recent earlier fact with that
    relation (visible to the agent), or None. Only event-pattern slots whose subject is a
    bare noun ("the deadline", "the launch") are candidates; resolution is recorded in the
    slot's provenance. Deterministic: depends only on earlier stored memories.
    """
    out = []
    for s in slots:
        src = s.get("source") or {}
        if src.get("rule") == "event" and s.get("relation") == "when" and s.get("entity"):
            noun = s["entity"]
            target = lookup(noun)
            if target and target != noun:
                s = {**s, "slot_id": f"{target}::{noun}", "entity": target, "relation": noun,
                     "source": {**src, "coreference": {"phrase": f"the {noun}", "resolved_to": target}}}
        out.append(s)
    return out


# Questions a fact answers, phrased the way people ask them. Used as extra retrieval keys
# (recall.question_keys): a question is far closer to another question in embedding space
# than to the statement that answers it. Templates only: deterministic, no model.
_QUESTION_TEMPLATES = {
    "location": ("Where does {s} live?", "Where is {s} based?"),
    "name": ("What is {p} name?", "What is {s} called?"),
    "age": ("How old is {s}?",),
    "birthday": ("When is {p} birthday?",),
    "allergy": ("What is {s} allergic to?",),
    "job": ("What does {s} do for work?", "What is {p} job?"),
    "employer": ("Where does {s} work?",),
    "email": ("What is {p} email address?",),
    "phone": ("What is {p} phone number?",),
    "deadline": ("When is {p} deadline?", "When is {s} due?"),
    "when": ("When is {s}?", "When does {s} happen?"),
    "launch": ("When does {s} launch?",),
    "alias_of": ("What else is {s} called?",),
}


def question_keys(slot, display_entity=None):
    """
    The questions a slot answers. Facts about the user are phrased the way the user would ask
    them (user::location -> "Where do I live?"); facts about a named entity use its name
    (atlas::launch -> "When does atlas launch?").
    """
    rel, ent = slot.get("relation", ""), slot.get("entity", "")
    if not rel or not ent:
        return []
    if ent == "user":
        first = {"location": ("Where do I live?",), "name": ("What is my name?",), "age": ("How old am I?",),
                 "birthday": ("When is my birthday?",), "allergy": ("What am I allergic to?",),
                 "job": ("What do I do for work?",), "employer": ("Where do I work?",),
                 "email": ("What is my email address?",), "phone": ("What is my phone number?",)}
        if rel in first:
            return list(first[rel])
        return [f"What is my {rel.replace('_', ' ')}?"]
    subject = display_entity or ent
    templates = _QUESTION_TEMPLATES.get(rel, ("What is {p} " + rel.replace("_", " ") + "?",))
    return [t.format(s=subject, p=f"{subject}'s") for t in templates]

