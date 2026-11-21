"""
Truth arbitration: at most one value per fact slot.

Candidates are grouped by the fact slot that answers the question (or, if
none, the memory's first fact). Within a slot, memories are walked in time
order: a newer value supersedes an older one, and an explicit negation of the
fact ("X is not Y", "X is no longer Y") invalidates it. Only the surviving
value of each slot is returned; memories without facts pass through.

Scores are not changed here; ordering is by the ranker's relevance.
"""
import json
import re
from collections import defaultdict

_NEGATED_FACT = re.compile(r"\b(?:is|are|was|were|am)\s+(?:not|no\s+longer|n't)\b|\b(?:isn't|aren't|wasn't)\b", re.I)
_NEGATED_VALUE = re.compile(r"^\s*(?:not|no\s+longer)\b", re.I)
_FILLER = {"anymore", "any", "more", "now", "longer", "the", "a", "an", "still"}


def _value_tokens(value):
    return {t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if t not in _FILLER}


def _same_value(a, b):
    """Values match if one's content tokens contain the other's (e.g. 'December 12' vs 'December 12 anymore')."""
    return bool(a) and bool(b) and (a <= b or b <= a)


class TruthEngine:
    def resolve(self, ranked):
        if not ranked:
            return []
        groups = defaultdict(list)
        free = []
        for mem in ranked:
            metadata = mem.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except ValueError:
                    metadata = {}
            slots = metadata.get("slots", []) if isinstance(metadata, dict) else []
            chosen = mem.get("_query_slot") or (slots[0] if slots else None)
            if chosen and chosen.get("slot_id"):
                mem["_slot_value"] = chosen.get("value")
                groups[chosen["slot_id"]].append(mem)
            else:
                mem["truth_status"] = "verified"
                free.append(mem)

        resolved = list(free)
        for slot_id, items in groups.items():
            active = None
            history = []  # every asserted (non-negation) memory of this slot, oldest first
            for m in sorted(items, key=lambda x: float(x.get("timestamp") or 0.0)):
                if _NEGATED_FACT.search(str(m.get("content", ""))) or _NEGATED_VALUE.match(str(m.get("_slot_value", ""))):
                    # "X is not V" invalidates the value V only. If V is the current
                    # value, fall back to the newest earlier value that is not V.
                    negated = _value_tokens(_NEGATED_VALUE.sub("", str(m.get("_slot_value", ""))))
                    m["truth_status"] = "negation"
                    if active is not None and _same_value(_value_tokens(active.get("_slot_value", "")), negated):
                        active["truth_status"] = "invalidated"
                        alive = [h for h in history if h.get("truth_status") != "invalidated"
                                 and not _same_value(_value_tokens(h.get("_slot_value", "")), negated)]
                        active = alive[-1] if alive else None
                    continue
                history.append(m)
                if active is not None and m.get("_slot_value") != active.get("_slot_value"):
                    active["truth_status"] = "superseded"
                    active["superseded_by"] = m.get("id")
                active = m
            if active is not None:
                active["truth_status"] = "verified"
                resolved.append(active)

        resolved.sort(key=lambda x: (x.get("is_hop_result", False), x.get("relevance", 0.0)), reverse=True)
        return resolved
