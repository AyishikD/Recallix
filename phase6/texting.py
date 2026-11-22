"""
Texting-style variants of labelled messages, for training and for a robustness check.

Fresh, independently written test data (eval/data/fresh, Gemini batch 1) showed how real users
text an assistant: all lowercase, subject dropped, contractions without apostrophes, fillers
("yar", "btw", "lol"). The rule grammar extracted a fact from 1 of 50 such messages, and the
teacher-labelled pool (Taskmaster, MultiWOZ, SGD, synthetic) is mostly tidy, capitalised text,
so a student trained on it alone would inherit the same blind spot.

texting_variant(text, facts, seed) rewrites a message in that style and moves every labelled
value span with it (the value goes through the same rewrite, then is found again in the new
text, nearest to its original relative position). If any value cannot be found again, it
returns None rather than a mislabelled example. No API calls, deterministic per seed.
"""
import random
import re

_CONTRACTIONS = [("i'm", "im"), ("i’m", "im"), ("don't", "dont"), ("don’t", "dont"), ("can't", "cant"),
                 ("can’t", "cant"), ("won't", "wont"), ("won’t", "wont"), ("it's", "its"), ("it’s", "its"),
                 ("i've", "ive"), ("i’ve", "ive"), ("i'll", "ill"), ("i’ll", "ill"), ("didn't", "didnt"),
                 ("didn’t", "didnt"), ("that's", "thats"), ("that’s", "thats")]
_SHORT = [(r"\byou\b", "u"), (r"\bplease\b", "pls"), (r"\bthanks\b", "thx"), (r"\btomorrow\b", "tmrw"),
          (r"\bbecause\b", "coz"), (r"\bokay\b", "ok")]
_PREFIX = ["", "", "btw ", "uff ", "so ", "ok so ", "hey "]
_SUFFIX = ["", "", " lol", " yar", " haha", " btw", " fyi"]


def _rewrite(s, rng_choices):
    """The deterministic part shared by message and value; rng_choices fixes the optional rewrites."""
    out = s.lower()
    for a, b in _CONTRACTIONS:
        out = out.replace(a, b)
    for (pat, rep), on in zip(_SHORT, rng_choices):
        if on:
            out = re.sub(pat, rep, out)
    return out


def texting_variant(text, facts, seed):
    """(new_text, new_facts) or None. facts: [{"relation", "value", "start", "end", "entity"?}]."""
    rng = random.Random(seed)
    choices = [rng.random() < 0.5 for _ in _SHORT]
    new = _rewrite(text, choices)
    new = re.sub(r"[.!]+\s*$", "", new) if rng.random() < 0.7 else new          # no final full stop
    if rng.random() < 0.5:
        new = re.sub(r"^i\s+(?=[a-z])", "", new)                                  # "i moved to pune" -> "moved to pune"
    new = rng.choice(_PREFIX) + new + rng.choice(_SUFFIX)
    out = []
    for f in facts:
        value = _rewrite(f["value"], choices).rstrip(".!")
        spots = [m.start() for m in re.finditer(re.escape(value), new)] if value else []
        if not spots:
            return None
        rel = f["start"] / max(1, len(text))
        start = min(spots, key=lambda p: abs(p / max(1, len(new)) - rel))
        out.append({**f, "value": new[start:start + len(value)], "start": start, "end": start + len(value)})
    return new, out
