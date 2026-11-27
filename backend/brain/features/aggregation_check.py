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
    from . import modes
    c = settings.get("recall.aggregation_check", {}) or {}
    mode = str(c.get("mode", "rescue")).lower()
    mode = mode if mode in ("rescue", "decide") else "rescue"
    helpful = modes.current() == "helpful"  # recall.mode helpful: the check on, in rescue mode
    return {"enabled": bool(c.get("enabled", False)) or helpful, "mode": "rescue" if helpful else mode,
            "model": c.get("model") or _DEFAULT_MODEL,
            "threshold": float(c.get("threshold", 0.5)), "memories": int(c.get("memories", 10)),
            "compute": bool(c.get("compute", True)), "compute_min_support": float(c.get("compute_min_support", 0.9)),
            "max_length": int(c.get("max_length", 256)), "device": c.get("device")}


# ---------------------------------------------------------------- question -> presupposed statements

_UNITS = r"(?:days?|weeks?|months?|years?|hours?|minutes?)"
# words that add nothing to the event itself
_FILLER = re.compile(r"\b(?:actually|finally|eventually|ever|really|first of all|in total|altogether|so far|overall|"
                     r"exactly|roughly|about how|in the end|personally|myself|successfully|officially)\b\s*", re.I)
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
                         r"me|him|her|them|us|it|abroad|home|\w+ly|[A-Z][\w'\-]*|['\"‘“].*)$")
_VERB_PREFIX = {"test", "binge", "bulk", "window", "speed", "house", "job", "road", "scuba", "sky", "ice", "rock", "bird"}


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
    t = re.sub(r" that (were|was|got|did|didn't)\b", r" \1", t)   # "flights of mine that were cancelled"
    if re.match(rf"^{_DET} ", t, re.I):
        t = re.sub(r"\s+(?:where|when|in which|at which)\b.*$", "", t)   # "the follow-up where my cream got adjusted"
    words = t.split()
    if len(words) > 1 and words[0].lower() in _VERB_PREFIX and words[1].lower().endswith("ing"):
        return _sentence("I was " + t[0].lower() + t[1:])       # "test driving the Kia Soul"
    w0 = words[0].lower()
    if w0.endswith("ing") and len(w0) > 4 and (w0 not in _NOT_GERUND or _is_gerund(words[0], words[1] if len(words) > 1 else None)):
        # "ordering the kitchen cabinets", "starting yoga". The person need not have done it with their own hands
        # ("loading the container": the movers did), so the event is also read without a doer.
        return (_sentence("I was " + t[0].lower() + t[1:]), _sentence(t + " happened"))
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


def _did_form(statement):
    """"I was signing the paperwork." -> ("I did sign the paperwork.", "I did signe the paperwork."): the base verb
    needs no tense table; both spellings are tried because "driving" may come from "drive"."""
    m = re.match(r"^(I|We|[A-Z][a-z]+(?: [a-z]+){0,2}) (?:was|were) (\w+?)ing\b(.*)$", statement or "")
    if not m or len(m.group(2)) < 2:
        return ()
    who, stem, rest = m.group(1), m.group(2), m.group(3).rstrip(".")
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
        stems = (stem[:-1],)                                    # "shopping" -> "shop"
    else:
        stems = (stem, stem + "e")                              # "signing" -> "sign"; "driving" -> "drive"
    return tuple(_sentence(f"{who} did {x}{rest}") for x in stems)


def _alternatives(claim):
    """
    A claim plus looser phrasings of it, as a tuple (one statement stays a string). "My second haircut
    happened" is also tried as "My haircut happened": which one was first or second is worked out across
    memories; no single memory says it.
    """
    variants = list(_both(claim, None) or ()) if isinstance(claim, tuple) else [claim]
    variants = [v for v in variants if isinstance(v, str)]
    for v in list(variants):
        variants.extend(_did_form(v))
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


_PRONOUN = r"\b(?:it|them|that one|(?:the|my|his|her|their) (?:(?:first|second|third|last|next|new|other) )?one)\b"


def _both(a, b):
    """Two readings of one statement (scored separately, the better counts); one when they are the same."""
    out = []
    for x in (a, b):
        for v in (x if isinstance(x, tuple) else (x,)):
            if v and v not in out:
                out.append(v)
    return None if not out else out[0] if len(out) == 1 else tuple(out)


def _resolve_it(second, first):
    """"... did I actually run it" after "I signed up for the Lisbon 10K" -> "the Lisbon 10K"."""
    if not re.search(_PRONOUN, second, re.I):
        return second
    starts = [m.start() for m in re.finditer(rf"\b{_DET}\b", first, re.I)]
    noun = first[starts[-1]:].strip(" .") if starts else ""
    if not noun or len(noun.split()) > 5 or re.search(r"\b(?:would|was|were|is|are|had|did|i)\b", noun, re.I):
        return second
    if re.search(r"\b(?:first|second|third|last|next|new|other) one\b", second, re.I):   # keep "second": "my second haircut"
        head = _ORDINAL.sub("", re.sub(rf"^{_DET} ", "", noun, flags=re.I)).strip()
        return re.sub(r"\bone\b", head, second, count=1, flags=re.I) if head else second
    return re.sub(_PRONOUN, noun, second, count=1, flags=re.I)


# who the question is about: "I", "we", a name ("John", "Mel and Dave"), a possessive ("Joanna's scripts",
# "my sister")
_SUBJ = (r"(i|we|(?-i:[A-Z][a-z]+)(?:'s \w+)?(?: and (?-i:[A-Z][a-z]+))?|(?:my|his|her|their|our) \w+(?: (?-i:[A-Z][a-z]+))?)")


def _mid(text):
    """An item moved from the start of the question into the middle of a statement: "Baby showers" -> "baby showers"."""
    words = text.split()
    if words and words[0][:1].isupper() and words[0][1:].islower() and \
            (re.match(rf"^{_DET}$", words[0], re.I) or (len(words) > 1 and words[1][:1].islower())):
        return words[0].lower() + text[len(words[0]):]
    return text


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
                 r"(?:first|last|earlier|later|next|second|most recently)(?: (?:in|of|during) [^,:]+)?[,:]?\s+(.+)$", q, re.I)
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


def _pair(body, between=False):
    """'A and B' (between ...) / 'A to B' / 'A until B' (from ...) -> (A, B), cutting where the second event starts."""
    body = re.sub(r",\s*(?:what|how)\b.*$", "", body, flags=re.I)   # "... to B, what was the duration in weeks?"
    link = r"(?:and)" if between else r"(?:to|until|till|and)"
    cut = re.search(rf",?\s+{link}\s+(?=(?:the (?:day|time|moment|arrival)|when|me |my |i |{_DET} |\w+ing\b))", body, re.I) \
        or re.search(rf",?\s+{link}\s+", body, re.I)
    return (body[:cut.start()], body[cut.end():]) if cut else None


def _span(text):
    """The two events after "between ..." / "from ..." in a question, or None."""
    m = re.search(r"\b(between|from) (.+)$", text, re.I)
    pair = m and _pair(m.group(2), between=m.group(1).lower() == "between")
    return _two_events(*pair) if pair else None


def _two_events(a, b):
    shared = re.match(r"^(.*?\b\w+ing) \S", _clean(a))            # "Maria adopting Coco and Shadow": Shadow too
    if shared and len(b.split()) <= 3 and not _FINITE.search(b) and not re.search(r"\w+ing\b", b):
        return [_event(a), _both(_event(f"{shared.group(1)} {b}"), _event(b))]
    return [_event(a), _both(_event(b), _event(_resolve_it(b, _clean(a))))]


def _then(a, b):
    """The second event of "... after A did <b>": b is "I actually run it" or "his new tablet arrive"."""
    b = _clean(b)
    s = re.match(r"^(i|we) (.+)$", b, re.I)
    if s:
        return _both(_did(s.group(2)), _did(_resolve_it(s.group(2), _clean(a))))
    # "his new tablet arrive": the subject ends where the base verb starts, which rules cannot
    # tell without a tagger, so each split is scored and the best counts
    words = _resolve_it(b, _clean(a)).split()
    return tuple(_sentence(" ".join(words[:i]) + " did " + " ".join(words[i:])) for i in range(1, min(len(words), 5))) or None


def _time_between(q):
    m = re.match(rf"^how (?:many {_UNITS}(?: and {_UNITS})?|long|much time)\b(.*)$", q, re.I)
    if not m:
        return None
    rest = _PAREN.sub("", m.group(1)).strip()
    rest = re.sub(r"^(?:(?:had |have |has )?(?:passed|elapsed|went by|gone by|were there|was there|was it|are there|is it|"
                  r"did it take|did i wait|apart were|separated)\s*)+", "", rest, flags=re.I)
    if re.match(r"^(?:between|from) ", rest, re.I) and _span(rest):
        return _span(rest)
    m = re.match(r"^(?:after|before|since|following) (.+?) (?:did|was|were|had) (.+)$", rest, re.I)
    if m:                                                       # "... after I signed up for X did I actually run it?"
        return [_event(m.group(1)), _then(m.group(1), m.group(2))]
    return None


# ---------------------------------------------------------------- freer phrasings
# The rules above read the textbook forms. People also front the topic ("Baby showers - how many have I
# attended?"), ask for a count without "how many" ("What's the number of ...", "Count the ...", "How often
# ..."), or put the events first ("I signed the lease - how many days later did I move in?"). These are
# reduced to: what kind of question, which phrases name the events, and which verb the question applies to them.

_BREAK = re.compile(r"\s+[-–—]\s+|:\s+")
_CUE = re.compile(r"\b(?:how many|how often|how long|how much time|which|what(?:'s| is| was)?|put|order|number of|gap|total)\b", re.I)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_PAST = rf"(?:\w{{2,}}ed|{_IRREGULAR_PAST}|been|gone|done|taken|seen|eaten|given|written|flown|driven|bought|run|won)"


def _normalise(q):
    q = re.sub(r",? please$", "", q.rstrip(" .!?"), flags=re.I)
    q = re.sub(r",? (?:with|without) [^,]+? (?:included|excluded|left out)\b", "", q, flags=re.I)
    q = re.sub(r"^(?:can you |could you )?(?:tell me|give me|do you know|remind me)\s+", "", q, flags=re.I)
    q = re.sub(rf"^in (?:{_UNITS})(?: \(?and {_UNITS}\)?)?,\s*", "", q, flags=re.I)             # "In weeks and days, how long ..."
    q = re.sub(r"^(?:including|counting|leaving|not counting|excluding|with|without) [^,]+,\s*", "", q, flags=re.I)
    q = re.sub(rf"\b(?:on )?{_MONTH} \d{{1,2}}(?:st|nd|rd|th)?,\s+", "", q, flags=re.I)        # "From June 7, my first ..."
    q = re.sub(rf"^on {_MONTH} \d{{1,2}}(?:st|nd|rd|th)?,? ", "", q, flags=re.I)
    # past only: "How often does John take his kids to the park?" asks for a habit, which one memory states
    q = re.sub(r"\bhow often(?= (?:did|have|has|had|was|were)\b)", "how many times", q, flags=re.I)
    q = re.sub(r"\b(?:what's|what is|what was) the (?:total )?number of\b", "how many", q, flags=re.I)
    q = re.sub(r"\b(?:what's|what is|what was) the (?:total )?number\b", "how many", q, flags=re.I)
    q = re.sub(r"^(?:count )?the (?:total )?number of\b", "how many", q, flags=re.I)
    q = re.sub(r"\b(?:separate |different |distinct )?(?:occasions|times|instances) (?:on which|when|that|where) ", "times ", q, flags=re.I)
    q = re.sub(r",\s*whether\b.*$", "", q, flags=re.I)                                       # "..., whether matcha or ramen bowls"
    m = re.match(r"^((?:for|at|in|with|to|on|during) [^,]+),\s*(how many .+)$", q, re.I)       # "For my brother, how many ..."
    if m:
        q = f"{m.group(2)} {m.group(1)[0].lower() + m.group(1)[1:]}"
    return q[:1].upper() + q[1:] if q else q


def _relative(text):
    """"weddings I've been to" -> "I have been to weddings."; "gifts I bought for my mom" -> "I bought gifts for my mom."."""
    m = re.match(r"^(?:the |my |all the )?(.+?),? (?:that |which )?\b(i|we)('ve| have| had)? (.+)$", text.strip(), re.I)
    if not m:
        return None
    thing, who, aux, rest = m.group(1), m.group(2), (m.group(3) or "").strip(" '"), _clean(m.group(4))
    if not rest or not re.match(rf"^(?:had |have )?{_PAST}\b", rest, re.I) and not aux:
        return None
    lead = re.match(r"^((?:had|have) )?(\w+)\s*(.*)$", rest)
    aux = "have" if aux == "ve" else aux or (lead.group(1) or "").strip()
    verb, tail = lead.group(2), lead.group(3)
    who = "I" if who.lower() in ("i", "we") else who
    thing = _mid(thing)
    if len(tail.split()) <= 2 and (not tail or re.search(r"\b(?:to|at|in|on|for|with|about|of|from|up|out)$", tail)):
        body = f"{verb} {tail} {thing}"                         # "been to" + weddings
    else:
        body = f"{verb} {thing} {tail}"                         # "bought" + gifts + "for my mom"
    return _sentence(" ".join(x for x in (who, aux, _clean(body)) if x))


def _applied(who, verb, item, base=False):
    """The question's own verb applied to one item: ("I", "finished", "'Deep Work'") -> "I finished 'Deep Work'."."""
    item = _clean(item)
    first = item.split()[0] if item else ""
    following = item.split()[1] if len(item.split()) > 1 else None
    doing = first.lower().endswith("ing") and len(first) > 4 and (first.lower() not in _NOT_GERUND or _is_gerund(first, following))
    if not verb or verb.lower() in ("did", "do", "done") or re.match(r"^(?:i|me) ", item, re.I) or doing:
        if verb and verb.lower() in ("did", "do", "done") and not doing and not re.match(r"^(?:i|me) ", item, re.I) \
                and not _np_like(item):
            return _state(who, "did", item)                     # "what did I do first: install the skylight or ..."
        return _event(item)                                     # the item is itself an event
    item = _mid(item)
    return _state(who, "did", f"{verb} {item}") if base else _sentence(f"{'I' if who.lower() in ('i', 'we') else who} {verb} {item}")


def _free_ordering(topic, ask, q):
    m = re.match(rf"^did {_SUBJ} (.+?) (?:before or after|after or before|before|after) (.+?)(?:,.*)?$", q, re.I)
    if m:                                                       # "Did I have the scan before or after starting the infusions?"
        return [_state(m.group(1), "did", _clean(m.group(2))), _event(m.group(3))]
    m = re.match(r"^(?:after|following) (.+?), which .*?\b(i|we) (\w+)$", q, re.I)   # "After 'Educated', which book was the next one I finished?"
    if m:
        return [_applied(m.group(2), m.group(3), m.group(1))]
    if topic is None:
        m = re.match(rf"^what (?:was|is) the order in which {_SUBJ} (\w+) (.+)$", q, re.I)
        if m:                                                   # "What was the order in which I sent A, B and C?"
            return [_applied(m.group(1), m.group(2), i) for i in _split_items(m.group(3))]
        m = re.match(r"^what (?:was|is) the order of (.+)$", q, re.I)
        if m:                                                   # "What was the order of me clearing the attic and cleaning the gutters?"
            items = _split_items(re.sub(r"^me ", "", m.group(1), flags=re.I))
            lead = re.match(r"^(\w+ing) ", items[0]) if items else None
            out = [_event(items[0])] if items else []
            for i in items[1:]:                                 # a later item may share the first one's verb
                out.append(_both(_event(i), _event(f"{lead.group(1)} {i}")) if lead and not _is_gerund(i.split()[0], "the") else _event(i))
            return out
        m = re.match(r"^between (.+?), which\b(.*)$", q, re.I)   # "Between A and B, which one came first?"
        if m:
            topic, ask = m.group(1), "which" + m.group(2)
        else:
            return None
    if re.match(r"^(?:i|we) ", topic, re.I) and re.search(r"\b(?:next|after that|afterwards)\b", ask, re.I):
        return [_event(topic)]                                  # "I finished 'Educated' - what's the next book I finished after that?"
    items = _split_items(topic)
    if not items:
        return None
    who, verb, base = "I", None, False
    m = re.search(rf"\bin (?:the )?order (?:that |in which )?{_SUBJ} (\w+)", ask, re.I)       # "put them in the order I finished them"
    if m:
        who, verb = m.group(1), m.group(2)
    else:
        m = re.search(rf"\bwhich (?:\w+ )?did {_SUBJ} (\w+)", ask, re.I)                      # "which car did I own earlier"
        if m:
            who, verb, base = m.group(1), m.group(2), True
        else:
            m = re.search(rf"\bwhich (?:was|were|got) ({_PAST}) \w+ by (me|us)\b", ask, re.I)  # "which was shipped first by me"
            if m:
                verb = m.group(1)
            elif re.search(r"\bwas mine\b", ask, re.I):                                       # "which of these cars was mine first"
                return [_sentence(f"{_clean(i)} was mine") for i in items]
    return [_applied(who, verb, i, base) for i in items]


def _free_time(topic, ask, q):
    if topic is not None:
        if re.match(r"^(?:from|between) ", topic, re.I) and _span(topic):   # "From A to B - how many days was that?"
            return _span(topic)
        m = re.search(r"\blater (?:did|was|were) (.+)$", ask, re.I)   # "A - how many days later did I B?"
        if m:
            return [_event(topic), _then(topic, m.group(1))]
        m = re.search(r"\bafter (.+?) (?:was|is|did) (?:that|it|this)(?: happen)?$", ask, re.I)   # "B - how many days after A was that?"
        if m:
            return [_event(m.group(1)), _event(topic)]
        both = re.split(r",?\s+and\s+(?:later |then |eventually |finally |afterwards )?", topic, maxsplit=1)
        if len(both) == 2:                                      # "A and later B - how many days apart were those?"
            a, b = both
            if re.match(rf"^(?:{_PAST})\b", _clean(b), re.I) and re.match(r"^(i|we) ", a, re.I):
                b = "I " + _clean(b)                            # "... and later got back from the meeting"
            return [_event(a), _both(_event(b), _event(_resolve_it(b, _clean(a))))]
        return None
    if _span(q):                                                # "How long was it between A and B?"
        return _span(q)
    m = re.search(r"\bhow long after (.+?) did (.+)$", q, re.I)
    if m:
        return [_event(m.group(1)), _then(m.group(1), m.group(2))]
    return None


def _free_counting(topic, ask, q):
    if topic is None:
        m = re.match(r"^(?:count|how many)(?: up)?(?: the| my)? times (?:that )?(.+)$", q, re.I)   # "Count the times I ..."
        if m and re.match(rf"^{_SUBJ} ", m.group(1), re.I):
            return [_event(m.group(1))]
        m = re.match(r"^how many times (was|were) (.+?) (\w+ed)$", q, re.I)   # "How often was an appointment of mine cancelled?"
        if m:
            return [_sentence(f"{m.group(2)} {m.group(1)} {m.group(3)}")]
        m = re.match(r"^(?:count(?: up)?|how many)\s+(.+)$", q, re.I)   # "Count the workshops I went to", "how many weddings I've been to"
        if m:                                                   # a bare "how many X does Y have" is one fact, not a count of events
            counted = q.lower().startswith("count")
            return [_relative(m.group(1)) or (_event(re.sub(r"^the ", "my ", m.group(1), flags=re.I)) if counted else None)]
        return None
    topic = re.sub(r"^count(?: up)?\s+", "", topic, flags=re.I)
    told = re.match(r"^(?:i|we) (?:told you about|mentioned|talked about|brought up) (.+)$", topic, re.I)
    if told:                                                    # "I told you about my brother's museum visits - how many of them?"
        return [_event(told.group(1))]
    m = re.match(rf"^how many(?: times)?(?: of them)? (did|have|has|had|do|does) {_SUBJ}\b(.*)$", ask, re.I)
    if m:                                                       # "Baby showers - how many have I attended?"
        aux, who, rest = m.group(1), m.group(2), _clean(m.group(3))
        rest = re.sub(r"\s*\b(?:it|that|them|those|this)$", "", rest).strip()
        if rest.lower() in ("do", "done", "did", ""):           # "Getting a flu shot - how many times have I done that?"
            return [_sentence("I was " + _mid(_clean(topic))) if topic.split()[0].lower().endswith("ing") else _event(topic)]
        if re.search(r"\bthere$", rest):                        # "Italy - how many times have I gone there?"
            return [_state(who, aux, re.sub(r"\bthere$", "to " + _mid(_clean(topic)), rest))]
        return [_state(who, aux, f"{rest} {_mid(_clean(topic))}")]
    m = re.match(rf"^how many {_SUBJ} ({_PAST})$", _clean(ask), re.I)   # "... - what's the number I actually had?"
    if m:
        return [_applied(m.group(1), m.group(2), topic)]
    return [_relative(topic) or _event(re.sub(r"^the ", "my ", topic, flags=re.I))]   # "Trips that I cancelled - how many were there?"


def _free(q):
    low = q.lower()
    if re.search(rf"how (?:long|much time)|how many {_UNITS}\b|\bgap\b", low):
        kind = "time_between"
    elif re.search(r"\border\b|\bbefore or after\b|\bwhich\b.*\b(?:first|earlier|earliest|later|latest|next|before|after)\b|"
                   r"\bwhat(?:'s| is| was)? .*\b(?:first|next)\b", low):
        kind = "ordering"
    elif re.search(r"\bhow many\b|\bcount\b|\btotal\b", low):
        kind = "counting"
    else:
        return None
    topic, ask = None, q
    parts = _BREAK.split(q, maxsplit=1)
    if len(parts) == 2:
        first, second = (x.strip(" ,.?") for x in parts)
        if _CUE.search(second) and not re.match(r"^(?:how|which|what|put)\b", first, re.I):
            topic, ask = first, second                          # "Baby showers - how many have I attended?"
        elif _CUE.search(first):
            topic, ask = second, first                          # "Which was earlier: A or B?"
    if topic is not None and not re.search(r"\b(?:it|that|them|those|there|these)\b", ask, re.I):
        for kind_, rule_ in (("time_between", _time_between), ("ordering", _ordering), ("counting", _counting)):
            claims = rule_(ask[:1].upper() + ask[1:])           # "home reno check - how many walls did I demolish?"
            if claims and all(claims):
                return kind_, claims
    rule = {"time_between": _free_time, "ordering": _free_ordering, "counting": _free_counting}[kind]
    claims = rule(topic, ask, q)
    return (kind, claims) if claims and all(claims) else None


def presuppositions(question):
    """
    {"kind": "counting" | "ordering" | "time_between", "claims": [...]} for a question answered by
    counting, ordering or subtracting across memories, else None. Each claim is a statement (or a tuple
    of alternative phrasings of one statement) that some memory must support.
    """
    q = re.sub(r"\s+", " ", str(question or "")).strip().rstrip("?.! ").strip()
    q = q.replace("’", "'")
    q = re.sub(r"^(?:apart from|besides|other than|excluding|not counting) [^,]+,\s*", "", q, flags=re.I)
    q = re.sub(r"^(?:hey|hi|ok|okay|so|and|also|by the way|quick question)[,:]?\s+", "", q, flags=re.I)
    found = None
    for text in dict.fromkeys((q, _normalise(q))):              # as asked, then with the wording normalised
        for kind, rule in (("time_between", _time_between), ("ordering", _ordering), ("counting", _counting)):
            claims = rule(text)
            if claims and all(claims):
                found = (kind, claims)
                break
        if found:
            break
    found = found or _free(_normalise(q))
    if found:
        claims = [_alternatives(c) for c in found[1]]
        if all(claims):
            return {"kind": found[0], "claims": claims}
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


# A memory that only records a plan ("booked flights to goa for march!!", "signed up for the marathon",
# "planning to visit Hampi next week") says the event was arranged, not that it happened. The entailment
# model reads "booked flights to Goa" as "flew to Goa", which answered "How many times did I fly to Goa?"
# for a trip that was cancelled. So a plan-only memory cannot support a claim that the event happened,
# unless the claim itself is about the plan ("How many days after booking the flights did I cancel them?").
_PLAN = re.compile(r"\b(?:book(?:ed|ing)?|reserv(?:ed|ing)|register(?:ed|ing)|sign(?:ed|ing) up|signed-up|schedul(?:ed|ing)|"
                   r"plan(?:ned|ning|s)?|intend(?:ed|ing)?|bought tickets|got tickets|tickets for|pre-?ordered)\b", re.I)
# "will", "going to" and the like only mark a plan when a future time is named ("I will never forget it" is not one)
_WEAK_PLAN = re.compile(r"\b(?:will|'ll|going to|gonna|hoping to|want to|thinking (?:of|about)|my next)\b", re.I)
_FUTURE = re.compile(r"\b(?:tomorrow|next (?:week|month|year|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
                     r"this (?:weekend|coming)|upcoming|soon|later this|in (?:a|two|three|few) (?:days|weeks|months)|"
                     r"(?:for|in|on|by) (?:the )?(?:\d{1,2}(?:st|nd|rd|th)?(?: of)? )?"
                     r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b)", re.I)
_DONE = re.compile(r"\b(?:today|tonight|this (?:morning|afternoon|evening)|just|finally|got back|came back|"
                   r"returned|yesterday|earlier today)\b", re.I)
_ABOUT_PLANS = re.compile(r"\b(?:book|reserv|regist|sign(?:ed|ing)? up|schedul|plan|cancel|postpon|resched|"
                          r"call(?:ed)? off|said i'?d|told you|intend)", re.I)


def plan_only(text):
    """True when a memory only records an arrangement for the future, not the event itself."""
    t = str(text or "")
    future = bool(_FUTURE.search(t))
    return (bool(_PLAN.search(t)) and (future or not _DONE.search(t))) or (bool(_WEAK_PLAN.search(t)) and future)


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
        best, best_memory, best_statement, discounted = 0.0, None, variants[0], None
        text_of = {m.get("id"): t for m, t in usable}
        support_by_memory = {}
        for (mid, statement), score in zip(owners, checker.entailment(pairs)):
            plan = not _ABOUT_PLANS.search(statement) and plan_only(text_of.get(mid))
            if not plan:
                support_by_memory[mid] = max(score, support_by_memory.get(mid, 0.0))
            if score > best and plan:
                discounted = discounted if discounted and discounted[1] >= score else (mid, score)
                continue
            if score > best:
                best, best_memory, best_statement = score, mid, statement
        entry = {"statement": best_statement, "support": round(best, 3), "memory": best_memory,
                 # every memory that supports the event (not only the best): computed answers need to know
                 # whether the event happened on one date or several
                 "supporters": {m: round(sc, 3) for m, sc in sorted(support_by_memory.items(), key=lambda x: -x[1])
                                if sc >= threshold}}
        ordinal = next((o for v in variants for o in ("first", "last") if re.search(rf"\b{o}\b", v, re.I)), None)
        if ordinal:
            entry["ordinal"] = ordinal
        if discounted and discounted[1] > best:
            entry["plan_only"] = {"memory": discounted[0], "support": round(discounted[1], 3)}
        report.append(entry)
    return {"kind": parsed["kind"], "supported": all(c["support"] >= threshold for c in report), "claims": report}


# ---------------------------------------------------------------- computed answers

def _day(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).date()


def _span_text(days, kind_hint=""):
    """41 -> "41 days (5 weeks 6 days)"; under a week just days; with whole months when 60 days or more."""
    out = f"{days} day{'s' if days != 1 else ''}"
    extra = []
    if days >= 7:
        w, d = divmod(days, 7)
        extra.append(f"{w} week{'s' if w != 1 else ''}" + (f" {d} day{'s' if d != 1 else ''}" if d else ""))
    if days >= 60:
        extra.append(f"about {days / 30.44:.1f} months")
    return out + (f" ({'; '.join(extra)})" if extra else "")


def compute(result, memories, min_support=0.9):
    """
    The answer worked out from the memories the check verified, or None. Only for time-between and ordering
    questions, and only when every presupposed event is supported at `min_support` by a distinct memory with a
    timestamp: the arithmetic is exact, choosing the events is the risk, so anything less is left to the caller.
    Counting is not computed (follow-ups, cancellations and near-misses make the set of occasions uncertain).
    {"kind", "text", "events": [{"statement", "memory", "date"}], "working"}.
    """
    if not result or not result.get("supported") or result.get("kind") not in ("time_between", "ordering"):
        return None
    claims = result.get("claims") or []
    if len(claims) < 2 or any(c.get("support", 0) < min_support or c.get("memory") is None for c in claims):
        return None
    events = []
    for c in claims:
        # every memory that supports this event strongly; several on different dates make it ambiguous,
        # unless the question says which one ("first", "last")
        cands = [m for m, sc in (c.get("supporters") or {c["memory"]: c["support"]}).items()
                 if sc >= min_support and memories.get(m) is not None and memories[m].get("timestamp") is not None]
        if not cands:
            return None
        by_date = sorted(cands, key=lambda m: memories[m]["timestamp"])
        if len({_day(memories[m]["timestamp"]) for m in cands}) > 1 and c.get("ordinal") not in ("first", "last"):
            return None
        pick = by_date[-1] if c.get("ordinal") == "last" else by_date[0]
        events.append({"statement": c["statement"], "memory": pick, "date": _day(memories[pick]["timestamp"])})
    if len({e["memory"] for e in events}) != len(events):
        return None
    if result["kind"] == "time_between":
        if len(events) != 2:
            return None
        a, b = events
        if b["date"] < a["date"]:
            return None  # "after A ... B" / "between A and B" with B dated before A: these are not the right events
        days = (b["date"] - a["date"]).days
        return {"kind": "time_between", "text": _span_text(days),
                "events": [{**e, "date": e["date"].isoformat()} for e in events],
                "working": f"{a['date'].isoformat()} (memory {a['memory']}) to {b['date'].isoformat()} (memory {b['memory']})"
                           f" = {days} days"}
    ordered = sorted(events, key=lambda e: e["date"])
    if len({e["date"] for e in events}) != len(events):
        return None  # two events on one day: the order is not known from dates
    return {"kind": "ordering", "text": " -> ".join(e["statement"].rstrip(".") for e in ordered),
            "events": [{**e, "date": e["date"].isoformat()} for e in ordered],
            "working": "; ".join(f"{e['date'].isoformat()}: {e['statement'].rstrip('.')} (memory {e['memory']})" for e in ordered)}
