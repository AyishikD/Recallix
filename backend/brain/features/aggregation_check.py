"""
Aggregation check: answer-or-decline for questions whose answer is worked out ACROSS memories.

"How many weddings did I attend?", "Which came first, the Tame Impala concert or my Denver hike?",
"How many days after I booked the flight did I arrive?" have no answer phrase inside any one
memory, so the extractive answer check (features/answer_check.py) cannot confirm them: on the
aggregation dev set it declined real questions and answered traps at nearly the same rate.

What such a question does have is PRESUPPOSITIONS: the events it takes for granted. "Which came
first, my knitting class or my yoga session?" presupposes a knitting class and a yoga session.
If the memories support every presupposed event, the question can be worked out from them; if one
event is supported by nothing (never mentioned, only planned and cancelled, someone else's), the
honest result is a decline.

  1. presuppositions(question): rewrite the question into one statement per presupposed event
     ("I did attend weddings." / "My knitting class happened." + "My yoga session happened.").
     Rules only; returns None for any question it does not recognise, which then takes the normal path.
  2. An entailment model (natural-language inference: premise = a memory, hypothesis = the statement;
     cross-encoder/nli-deberta-v3-xsmall, 22M-parameter encoder, Apache 2.0, CPU) scores each
     statement against each ranked memory. No LLM.
  3. Every statement entailed by some memory -> answer (the memories are served and the result says the
     answer still has to be worked out). Otherwise decline.

Configuration (config.yaml recall.aggregation_check): enabled, model, threshold, memories (how many
ranked memories are read), max_length, device. How it is applied: RecallEngine.multi_stage_recall.
"""
import re
import threading

from backend import settings

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def config():
    c = settings.get("recall.aggregation_check", {}) or {}
    return {"enabled": bool(c.get("enabled", False)), "model": c.get("model") or _DEFAULT_MODEL,
            "threshold": float(c.get("threshold", 0.5)), "memories": int(c.get("memories", 10)),
            "max_length": int(c.get("max_length", 256)), "device": c.get("device")}


# ---------------------------------------------------------------- question -> presupposed statements

_UNITS = r"(?:days?|weeks?|months?|years?|hours?|minutes?)"
# words that add nothing to the event itself
_FILLER = re.compile(r"\b(?:actually|finally|eventually|ever|really|first of all|in total|altogether|so far|overall|"
                     r"exactly|roughly|about how|in the end)\b\s*", re.I)
# trailing time qualifiers: a memory's text rarely states them (the date is the message's timestamp)
_TIME_TAIL = re.compile(
    r"\s+(?:(?:in|during|since|throughout|over) (?:the (?:past|last|whole) )?(?:\d{4}|this year|last year|the year|"
    r"the summer|the winter|the spring|the autumn|the fall|the past \w+ \w+)"
    r"|since we started (?:chatting|talking)|this year|last year|so far(?: this year)?"
    r"|(?:on|by|as of|until|before|up to) (?:the )?(?:\d{1,2}(?:st|nd|rd|th)? )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?: \d{1,2}(?:st|nd|rd|th)?)?(?:,? \d{4})?"
    r"|on \d{4}-\d{2}-\d{2})\s*$", re.I)
_PAREN = re.compile(r"\s*\([^)]*\)")
_LEAD = re.compile(r"^(?:the (?:day|time|moment|date)(?: (?:that|when|on which))?|when|the day of)\s+", re.I)
_DET = r"(?:the|a|an|my|our|his|her|their|this|that|these|those)"
_IRREGULAR_PAST = ("got|took|ran|went|met|sent|bought|sold|began|did|had|made|told|saw|flew|drove|left|quit|came|gave|"
                   "found|heard|wrote|read|paid|spent|won|lost|broke|built|caught|chose|drew|ate|fell|felt|forgot|grew|"
                   "held|kept|knew|led|put|rode|said|sang|sat|slept|spoke|stood|swam|taught|thought|threw|woke|wore|"
                   "was|were|became|brought|hit|cut|set|let|hurt|shut|hung|rang|rose|shot|stole|struck|swore|tore")
_FINITE = re.compile(rf"\b(?:\w{{2,}}ed|{_IRREGULAR_PAST})\b", re.I)
_ING = re.compile(r"^(\w{2,}ing)\b", re.I)
_NOT_GERUND = {"morning", "evening", "wedding", "meeting", "building", "painting", "training", "booking", "housewarming",
               "thing", "something", "anything", "nothing", "everything", "king", "ring", "spring", "string", "ceiling",
               "sibling", "darling", "pudding", "clothing", "gathering", "opening", "outing", "screening", "tasting",
               "listing", "hearing", "viewing", "fitting", "reading", "writing", "recording", "briefing", "hiking"}


def _clean(text):
    t = _PAREN.sub("", str(text or ""))
    t = _FILLER.sub("", t)
    prev = None
    while prev != t:
        prev, t = t, _TIME_TAIL.sub("", t)
    return re.sub(r"\s+", " ", t).strip(" ,.?;:")


def _sentence(text):
    t = re.sub(r"\s+", " ", text).strip(" ,.?;:")
    return (t[0].upper() + t[1:] + ".") if t else None


_AFTER_VERB = re.compile(rf"^(?:{_DET}|to|at|in|on|for|with|from|into|up|out|off|over|back|down|away|about|through|"
                         r"me|him|her|them|us|it|[A-Z][\w'\-]*|['\"‘“].*)$")


def _is_gerund(word, following=None):
    """
    "moving" in "my sister moving to Davao" is the event's verb; "gardening" in "my first gardening
    session" is part of a name. Without a tagger the next word decides: a verb is followed by its
    object or a preposition. Returns True, False, or None when it cannot tell (the phrase ends there).
    """
    w = (word or "").lower()
    if not (w.endswith("ing") and len(w) > 4):
        return False
    if following is None:
        return None if w not in _NOT_GERUND else False
    return bool(_AFTER_VERB.match(following))


def _event(phrase):
    """
    One presupposed event, as a statement. The phrase is whatever the question used to name it:
    a clause ("I signed the lease", "my friend Liza got her haircut"), a gerund ("ordering the
    cabinets", "me adopting Milo", "my sister moving to Davao") or a noun phrase ("my Denver hike",
    "the Tame Impala concert", "the workshop I attended").
    """
    t = _LEAD.sub("", _clean(phrase))
    t = _LEAD.sub("", t)
    if not t:
        return None
    low = t.lower()
    if low.startswith("i "):                                    # "I signed the lease"
        return _sentence("I " + t[2:])
    m = re.match(r"^me (\w+ing)\b(.*)$", t, re.I)               # "me adopting Milo"
    if m:
        return _sentence(f"I was {m.group(1)}{m.group(2)}")
    words = t.split()
    if _is_gerund(words[0], words[1] if len(words) > 1 else None):   # "ordering the kitchen cabinets"
        return _sentence("I was " + t[0].lower() + t[1:])
    m = re.match(rf"^({_DET} .+?) i (\w+)\b(.*)$", t, re.I)     # "the workshop I attended"
    if m and " " not in m.group(2):
        return _sentence(f"I {m.group(2)} {m.group(1)[0].lower() + m.group(1)[1:]}{m.group(3)}")
    m = re.match(rf"^({_DET} .+?) ([A-Z][a-z]+) ((?:{_IRREGULAR_PAST}|\w+ed)(?: (?:to|at|in|on|for|with))?)$", t)
    if m:                                                       # "the show Alex went to"
        return _sentence(f"{m.group(2)} {m.group(3)} {m.group(1)[0].lower() + m.group(1)[1:]}")
    if _FINITE.search(" ".join(words[1:])):                     # "my friend Liza got her haircut"
        return _sentence(t)
    happened = _sentence(t + " happened")                       # "my Denver hike", "the Tame Impala concert"
    for i, w in enumerate(words[1:], 1):                        # "my sister moving to Davao", "the new phone arriving"
        if i > 5:
            break
        g = _is_gerund(w, words[i + 1] if i + 1 < len(words) else None)
        if g or g is None:
            was = _sentence(" ".join(words[:i]) + " was " + " ".join(words[i:]))
            return was if g else (happened, was)                # "my phone screen cracking" / "my first dentist cleaning"
    return happened


_ORDINAL = re.compile(r"\b(?:first|second|third|fourth|fifth|initial|final|last|next|latest|earliest|most recent)\s+", re.I)


def _alternatives(claim):
    """
    A claim plus looser phrasings of it, as a tuple (one statement stays a string). "My second haircut
    happened" is also tried as "My haircut happened": which one was first or second is worked out across
    memories; no single memory says it.
    """
    variants = list(claim) if isinstance(claim, tuple) else [claim]
    for v in list(variants):
        if v and _ORDINAL.search(v):
            variants.append(_sentence(_ORDINAL.sub("", v)))
    variants = list(dict.fromkeys(v for v in variants if v))
    return variants[0] if len(variants) == 1 else tuple(variants) if variants else None


def _split_items(text):
    """'A, B and C' / 'A or B' -> ['A', 'B', 'C']."""
    parts = re.split(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", text)
    return [p.strip(" ,.?;:") for p in parts if p and p.strip(" ,.?;:")]


def _np_like(item):
    w = item.split()[0] if item.split() else ""
    return bool(re.match(rf"^{_DET}$", w, re.I)) or w[:1].isupper() or w[:1] in "'\"‘“" or w[:1].isdigit() or len(item.split()) == 1


def _did(verb_and_rest):
    """Do-support keeps the base verb, so no tense table is needed: "I did attend weddings"."""
    return _sentence("I did " + verb_and_rest)


_PRONOUN = r"\b(?:it|them|that one|the (?:(?:first|second|third|last|next|new|other) )?one)\b"


def _both(a, b):
    """Two readings of one statement (scored separately, the better counts); one when they are the same."""
    return a if a == b or not b else b if not a else (a, b)


def _resolve_it(second, first):
    """"... did I actually run it" after "I signed up for the Lisbon 10K" -> "the Lisbon 10K"."""
    if not re.search(_PRONOUN, second, re.I):
        return second
    starts = [m.start() for m in re.finditer(rf"\b{_DET}\b", first, re.I)]
    noun = first[starts[-1]:].strip(" .") if starts else ""
    if not noun or len(noun.split()) > 5 or re.search(r"\b(?:would|was|were|is|are|had|did|i)\b", noun, re.I):
        return second
    return re.sub(_PRONOUN, noun, second, count=1, flags=re.I)


# who the question is about: "I", "we", a name ("John", "Mel and Dave"), a possessive ("Joanna's scripts",
# "my sister")
_SUBJ = (r"(i|we|(?-i:[A-Z][a-z]+)(?:'s \w+)?(?: and (?-i:[A-Z][a-z]+))?|(?:my|his|her|their|our) \w+(?: (?-i:[A-Z][a-z]+))?)")


def _state(subject, aux, rest):
    """("I", "did", "attend weddings") -> "I did attend weddings."; ("John", "has", "injured his ankle")."""
    me = subject.lower() in ("i", "we")
    aux = aux.lower()
    aux = "did" if aux in ("did", "do", "does") else ("have" if me else "has") if aux in ("have", "has") else aux
    return _sentence(f"{'I' if me else subject} {aux} {rest}")


def _counting(q):
    told = r"(?:tell you|told you|mention(?:ed)?|say|said|talk(?:ed)? to you)"
    m = re.match(rf"^how many times (?:did|have|had) i {told} (about |that |having )?(.+)$", q, re.I)
    if m:                                                       # the claim is what was told, not the telling
        return [_sentence("I had " + _clean(m.group(2)))] if (m.group(1) or "").strip() == "having" else [_event(m.group(2))]
    m = re.match(rf"^how many (?:of )?(.+?) (?:did|have|had) i {told}(?: about| of)?$", q, re.I)
    if m:
        return [_event(m.group(1))]
    m = re.match(rf"^how many times (did|have|has|had|do|does) {_SUBJ} (.+)$", q, re.I)
    if m:                                                       # "How many times did I go to Italy?"
        return [_state(m.group(2), m.group(1), _clean(m.group(3)))]
    m = re.match(rf"^how many (?:different |separate |distinct |total |other )?(.+?) (did|have|has|had|do|does) {_SUBJ} (\w+)\b(.*)$",
                 q, re.I)
    if m and not re.match(rf"^{_UNITS}\b", m.group(1), re.I) and m.group(4).lower() not in ("have", "own", "need", "want"):
        things, aux, subject, verb, rest = re.sub(r"^of ", "", m.group(1), flags=re.I), m.group(2), m.group(3), m.group(4), m.group(5)
        rest = re.sub(r"\s+that (?:were|was|are|is)\b.*$", "", rest, flags=re.I).strip()
        lead = re.match(r"^(\w+ing|on|to|in|at|up|out)\b\s*(.*)$", rest) if rest else None
        if lead and (lead.group(1).endswith("ing") or not lead.group(2)):   # "finish reading" + books, "been on" + hikes
            verb, rest = f"{verb} {lead.group(1)}", lead.group(2)
        if len(rest.split()) <= 3 and re.search(r"\b(?:about|of|for|to|with|at|in|on|from)$", _clean(rest)):
            body = _clean(f"{verb} {rest} {things}")            # "tell you about" + my brother's museum visits
        else:
            body = _clean(f"{verb} {things} {rest}")
        return [_state(subject, aux, body)]
    m = re.match(r"^how many (?:of )?(.+?) (were|was|got|did(?:n't| not)?|have|has|are|is) (.+)$", q, re.I)
    if m and not re.match(rf"^{_UNITS}\b", m.group(1), re.I) and not re.match(r"^times\b", m.group(1), re.I):
        return [_sentence(_clean(f"Some of {m.group(1)} {m.group(2)} {m.group(3)}"))]
    return None


def _ordering(q):
    m = re.match(rf"^which (?:(?:one|of (?:these|them|the two)) )?(?:(\w+(?: \w+)?) )?did {_SUBJ} (\w+)(?: \w+)? "
                 r"(?:first|last|earlier|later|next|second|most recently)[,:]?\s+(.+)$", q, re.I)
    if m:                                                       # "Which city did I visit first, New York or Boston?"
        noun, who, verb, items = m.group(1), m.group(2), m.group(3), _split_items(m.group(4))
        if noun:
            items = [re.sub(r"^the one\b", "the " + noun, i, flags=re.I) for i in items]
        if len(items) >= 2:
            return [_state(who, "did", _clean(f"{verb} {i}")) for i in items]
    m = re.match(rf"^which (?:\w+(?: \w+)? )?did {_SUBJ} (\w+)\b.*?\b(?:after|before|following|prior to) (.+)$", q, re.I)
    if m:                                                       # "Which book did I finish next after 'Educated'?"
        return [_state(m.group(1), "did", _clean(f"{m.group(2)} {m.group(3)}"))]
    m = re.match(r"^(?:which|what) (?:\w+ )?(?:came|happened|was|occurred|took place|did i do) "
                 r"(?:first|later|earlier|last|second|most recently|earliest|latest)[,:]?\s+(.+)$", q, re.I)
    if m:                                                       # "Which came first, A or B?"
        items = [i for i in re.split(r"\s+or\s+|\s*,\s*(?:or\s+)?", m.group(1)) if i.strip(" ,.?")]
        if len(items) >= 2:
            return [_event(i) for i in items]
    m = re.match(rf"^in what order did {_SUBJ} (?:first |then |originally )?(\w+) (?:the following \w+[:,]?\s*)?(.+)$", q, re.I)
    if m:                                                       # "In what order did I install A and tile B?"
        who, verb, items = m.group(1), m.group(2), _split_items(m.group(3))
        _did = lambda body: _state(who, "did", body)            # noqa: E731 (the question's own subject)
        if items:
            out = [_did(_clean(f"{verb} {items[0]}"))]
            # "have meals with Maya, Laila and Karim": the later names share "meals with"
            shared = re.match(r"^(.+\b(?:with|on|at|in|for|from|to|of|by)) \S", items[0], re.I)
            for i in items[1:]:
                if shared and not re.search(r"\b(?:with|on|at|in|for|from|to|of|by)\b", i, re.I) and \
                        not re.match(rf"^{_DET}\b", i, re.I) and (i[:1].isupper() or len(i.split()) == 1):
                    out.append((_did(_clean(f"{verb} {shared.group(1)} {i}")), _did(_clean(f"{verb} {i}"))))
                    continue
                # a later item is a noun phrase sharing the verb ("... and the Forum") or its own verb phrase
                # ("... and tile the bathroom floor"); when unclear both readings are scored and the better counts
                out.append(_did(_clean(f"{verb} {i}")) if _np_like(i) else (_did(_clean(i)), _did(_clean(f"{verb} {i}"))))
            return out
    return None


def _time_between(q):
    m = re.match(rf"^how (?:many {_UNITS}(?: and {_UNITS})?|long|much time)\b(.*)$", q, re.I)
    if not m:
        return None
    rest = _PAREN.sub("", m.group(1)).strip()
    rest = re.sub(r"^(?:(?:had |have |has )?(?:passed|elapsed|went by|gone by|were there|was there|was it|are there|is it|"
                  r"did it take|did i wait|apart were)\s*)+", "", rest, flags=re.I)
    m = re.match(r"^(?:between|from) (.+)$", rest, re.I)
    if m:
        body = m.group(1)
        cut = re.search(rf"\s+(?:and|to|until|till)\s+(?=(?:the (?:day|time|moment)|when|me |my |i |{_DET} |\w+ing\b))", body, re.I) \
            or re.search(r"\s+(?:and|to|until|till)\s+", body, re.I)
        if cut:
            a, b = body[:cut.start()], body[cut.end():]
            shared = re.match(r"^(.*?\b\w+ing) \S", _clean(a))    # "Maria adopting Coco and Shadow": Shadow too
            if shared and len(b.split()) <= 3 and not _FINITE.search(b) and not re.search(r"\w+ing\b", b):
                return [_event(a), (_event(f"{shared.group(1)} {b}"), _event(b))]
            return [_event(a), _both(_event(b), _event(_resolve_it(b, _clean(a))))]
    m = re.match(r"^(?:after|before|since|following) (.+?) (?:did|was|were|had) (.+)$", rest, re.I)
    if m:                                                       # "... after I signed up for X did I actually run it?"
        a, b = m.group(1), _clean(m.group(2))
        s = re.match(r"^(i|we) (.+)$", b, re.I)
        if s:
            second = _both(_did(s.group(2)), _did(_resolve_it(s.group(2), _clean(a))))
        else:
            # "his new tablet arrive": the subject ends where the base verb starts, which rules cannot
            # tell without a tagger, so each split is scored and the best counts
            words = b.split()
            second = tuple(_sentence(" ".join(words[:i]) + " did " + " ".join(words[i:]))
                           for i in range(1, min(len(words), 5))) or None
        return [_event(a), second]
    return None


def presuppositions(question):
    """
    {"kind": "counting" | "ordering" | "time_between", "claims": [...]} for a question answered by
    counting, ordering or subtracting across memories, else None. Each claim is a statement (or a tuple
    of alternative phrasings of one statement) that some memory must support.
    """
    q = re.sub(r"\s+", " ", str(question or "")).strip().rstrip("?").strip()
    q = q.replace("’", "'")
    q = re.sub(r"^(?:apart from|besides|other than|excluding|not counting) [^,]+,\s*", "", q, flags=re.I)
    q = re.sub(r"^(?:hey|hi|ok|okay|so|and|also|by the way|quick question)[,:]?\s+", "", q, flags=re.I)
    for kind, rule in (("time_between", _time_between), ("ordering", _ordering), ("counting", _counting)):
        claims = rule(q)
        if claims and all(claims):
            claims = [_alternatives(c) for c in claims]
            if all(claims):
                return {"kind": kind, "claims": claims}
    return None


# ---------------------------------------------------------------- entailment

class EntailmentChecker:
    """Scores (memory, statement) pairs with an NLI cross-encoder: P(the memory entails the statement)."""
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, model_name, max_length=256, device=None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        labels = {str(v).lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"{model_name} is not an NLI model (labels: {sorted(labels)})")
        self._entail = labels["entailment"]
        self._infer_lock = threading.Lock()

    @classmethod
    def from_config(cls):
        """The configured checker, or None if disabled or unavailable (logged once)."""
        cfg = config()
        if not cfg["enabled"]:
            return None
        key = (cfg["model"], cfg["max_length"], cfg["device"])
        with cls._lock:
            if key not in cls._instances:
                try:
                    cls._instances[key] = cls(*key)
                except Exception as e:  # missing sentence-transformers/torch, no network for the first download, ...
                    print(f"[AggregationCheck] disabled: cannot load {cfg['model']}: {e}")
                    cls._instances[key] = None
            return cls._instances[key]

    def entailment(self, pairs):
        if not pairs:
            return []
        with self._infer_lock:
            scores = self.model.predict(list(pairs), apply_softmax=True, show_progress_bar=False)
        return [float(row[self._entail]) for row in scores]


def _as_speaker(statement, speaker):
    """"John has injured his ankle." as John himself would say it: "I have injured my ankle." (or None)."""
    if not speaker or not re.search(rf"\b{re.escape(speaker)}\b", statement):
        return None
    name = re.escape(speaker)
    t = re.sub(rf"\b{name}'s\b", "my", statement)
    t = re.sub(rf"\b{name} has\b", "I have", t)
    t = re.sub(rf"\b{name} is\b", "I am", t)
    t = re.sub(rf"\b{name} was\b", "I was", t)
    t = re.sub(rf"^{name}\b", "I", t)
    t = re.sub(rf"\b{name}\b", "me", t)
    if t.startswith("I ") or t.startswith("My ") or " my " in t:
        t = re.sub(r"\b(?:his|her|their)\b", "my", t)
    return _sentence(t)


def check(checker, question, memories, threshold, read=None, speaker=None):
    """
    None when the question is not an aggregation question. Otherwise
    {"kind", "supported": bool, "claims": [{"statement", "support", "memory"}]}: for each presupposed
    event, the best entailment probability over the memories and the id of that memory.
    `read(memory) -> text` lets the caller decide how a memory is read (default: its content);
    `speaker(memory) -> name` lets a statement about that person also be read in the first person
    ("John has injured his ankle" against John's own "I twisted my ankle again").
    """
    parsed = presuppositions(question)
    if parsed is None or not memories:
        return None
    texts = [(read(m) if read else m.get("content", "")) or "" for m in memories]
    usable = [(m, t) for m, t in zip(memories, texts) if t.strip()]
    report = []
    for claim in parsed["claims"]:
        variants = claim if isinstance(claim, tuple) else (claim,)
        pairs, owners = [], []
        for statement in variants:
            for m, t in usable:
                for hyp in (statement, _as_speaker(statement, speaker(m)) if speaker else None):
                    if hyp:
                        pairs.append((t, hyp))
                        owners.append((m.get("id"), statement))
        best, best_memory, best_statement = 0.0, None, variants[0]
        for (mid, statement), score in zip(owners, checker.entailment(pairs)):
            if score > best:
                best, best_memory, best_statement = score, mid, statement
        report.append({"statement": best_statement, "support": round(best, 3), "memory": best_memory})
    return {"kind": parsed["kind"], "supported": all(c["support"] >= threshold for c in report), "claims": report}
