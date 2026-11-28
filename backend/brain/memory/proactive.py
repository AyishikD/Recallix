"""
Proactive recall: memories an agent should keep in mind for a new message, without a question.

Agents rarely know *when* to look something up: "book a table at the Thai place" does not ask
"what am I allergic to?", but the allergy matters. Recallic maps situations in the message to
the kinds of facts that matter there, then returns this user's current facts of those kinds
(visible to the asking agent), each with the reason it was surfaced.

Deterministic and LLM-free: the situation table below is plain, documented and editable
(config recall.proactive.situations overrides it). Facts that were later changed are not
surfaced in their old form — only each fact's current value.
"""
import re

from backend import settings

# situation -> (cue words, fact relations that matter in it)
SITUATIONS = {
    "eating or ordering food": (
        r"restaurant|dinner|lunch|breakfast|brunch|menu|order(?:ing)?|takeaway|takeout|cafe|caf[eé]|bakery|recipe|cook(?:ing)?|"
        r"bake|snack|meal|eat(?:ing)?|food|pizza|sushi|thai|indian|chinese|italian|mexican|burger|dessert|catering|party",
        ["allergy", "diet", "dislikes", "likes"]),
    "travel": (
        r"flight|fly(?:ing)?|trip|travel(?:ling|ing)?|hotel|airbnb|visa|passport|airport|vacation|holiday|abroad|booking a stay",
        ["destination", "travel_date", "budget", "allergy", "diet", "nationality"]),
    "health and appointments": (
        r"doctor|dentist|clinic|hospital|pharmacy|prescription|medicine|medication|appointment|check-?up|surgery|vaccin",
        ["allergy", "medication", "health", "event_time"]),
    "gifts and shopping": (
        r"gift|present|buy(?:ing)?|shop(?:ping)?|birthday|anniversary|surprise",
        ["likes", "dislikes", "birthday", "budget", "allergy"]),
    "scheduling": (
        r"meeting|schedule|calendar|reschedul|tomorrow|next week|this week|deadline|due|remind",
        ["event_time", "deadline", "when", "plan"]),
}


def situations():
    override = (settings.get("recall.proactive", {}) or {}).get("situations")
    return override or SITUATIONS


def detect(message):
    """[(situation, relations)] whose cue words occur in the message."""
    out = []
    for name, (cues, relations) in situations().items():
        if re.search(rf"\b(?:{cues})\b", message, re.I):
            out.append((name, relations))
    return out


def relevant(episodic, user_id, message, agent_id, limit=5):
    """[{"memory_id", "content", "fact", "value", "why"}] — current facts that matter for `message`."""
    import sqlite3
    found = detect(message)
    if not found:
        return []
    wanted = {}
    for name, relations in found:
        for rel in relations:
            wanted.setdefault(rel, name)
    conn = sqlite3.connect(episodic.db_path, timeout=30)
    try:
        rows = conn.execute(
            "SELECT e.id, e.content, e.timestamp, json_extract(j.value, '$.slot_id'), json_extract(j.value, '$.relation'), "
            "json_extract(j.value, '$.value') FROM episodic_events e, json_each(e.metadata, '$.slots') j "
            "WHERE e.user_id = ? AND e.status = 'ACTIVE' AND (e.memory_type = 'shared' OR e.agent_id = ?) "
            "ORDER BY e.timestamp, e.id", (user_id, agent_id)).fetchall()
    finally:
        conn.close()
    latest = {}  # slot -> newest row: a fact that later changed is surfaced only in its current form
    for mid, content, ts, slot, rel, value in rows:
        latest[slot] = (mid, content, ts, rel, value)
    out = []
    for slot, (mid, content, ts, rel, value) in sorted(latest.items(), key=lambda kv: -kv[1][2]):
        base = rel.split("_")[0] if rel not in wanted else rel  # learned multi-valued slots: likes_tea -> likes
        situation = wanted.get(rel) or wanted.get(base)
        if situation and mid not in {o["memory_id"] for o in out}:
            out.append({"memory_id": mid, "content": content, "fact": slot, "value": value,
                        "why": f"the message is about {situation}; this is a known {base.replace('_', ' ')}"})
    return out[:limit]
