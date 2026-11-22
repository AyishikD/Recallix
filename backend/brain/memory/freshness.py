"""
Fact half-lives: how likely a remembered fact is to still be true (survival analysis, no LLM).

Each fact slot (e.g. user::location, projectalpha::deadline) has a value history in the
user's own memories. Every time a value was replaced by a different one, we observe a
lifetime; a value that is still current has lasted *at least* this long (right-censored).
Per relation r (location, deadline, job, ...), with an exponential lifetime model:

    events_r   = number of observed replacements
    exposure_r = total time values of relation r were in force (replaced or still current)
    rate_r     = (a0 + events_r) / (b0 + exposure_r)          per day

which is the posterior mean of the replacement rate under a Gamma(a0, b0) prior — the
prior keeps a single early update from swinging the estimate (defaults a0 = 1 event per
b0 = 365 days, i.e. a prior half-life of about 253 days; config recall.freshness).
Then for a value set `age` days ago:

    P(still true) = exp(-rate_r * age),     half-life = ln 2 / rate_r

A value later replaced by a different one is superseded (probability 0). Restating the
same value renews it. Only this user's facts visible to the asking agent are used: no
other tenant's data. Time-dependent by nature, so `now` is explicit.
"""
import math
import sqlite3

DAY = 86400.0


def slot_history(db_path, user_id, agent_id):
    """[(slot_id, relation, value, timestamp)] of this user's facts visible to `agent_id`, oldest first."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        return conn.execute(
            "SELECT json_extract(j.value, '$.slot_id'), json_extract(j.value, '$.relation'), "
            "json_extract(j.value, '$.value'), e.timestamp "
            "FROM episodic_events e, json_each(e.metadata, '$.slots') j "
            "WHERE e.user_id = ? AND e.status = 'ACTIVE' AND (e.memory_type = 'shared' OR e.agent_id = ?) "
            "ORDER BY e.timestamp, e.id", (user_id, agent_id)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _norm(v):
    return " ".join(str(v or "").lower().split())


def _runs(history):
    """{slot_id: (relation, [(value, set_at)])}, consecutive restatements of a value merged (renewing it)."""
    runs = {}
    for slot_id, relation, value, ts in history:
        rel, seq = runs.setdefault(slot_id, (relation, []))
        if seq and _norm(seq[-1][0]) == _norm(value):
            seq[-1] = (seq[-1][0], ts)  # restated: renewed at the later time
        else:
            seq.append((value, ts))
    return runs


def fit_rates(history, now, a0=1.0, b0_days=365.0):
    """{relation: {"rate_per_day", "half_life_days", "events", "exposure_days"}}."""
    events, exposure = {}, {}
    for relation, seq in _runs(history).values():
        for i, (_, set_at) in enumerate(seq):
            end = seq[i + 1][1] if i + 1 < len(seq) else now
            exposure[relation] = exposure.get(relation, 0.0) + max(0.0, end - set_at) / DAY
            if i + 1 < len(seq):
                events[relation] = events.get(relation, 0) + 1
    out = {}
    for relation in set(exposure) | set(events):
        rate = (a0 + events.get(relation, 0)) / (b0_days + exposure.get(relation, 0.0))
        out[relation] = {"rate_per_day": rate, "half_life_days": math.log(2) / rate,
                         "events": events.get(relation, 0), "exposure_days": exposure.get(relation, 0.0)}
    return out


def assess(slot_id, relation, value, runs, rates, now):
    """{"still_true", "half_life_days", "superseded", "since"} for one fact; `runs` from _runs(history)."""
    _, seq = runs.get(slot_id, (relation, []))
    current = seq[-1] if seq else None
    if current is not None and _norm(current[0]) != _norm(value):
        return {"still_true": 0.0, "half_life_days": None, "superseded": True, "since": None,
                "replaced_by": current[0]}
    set_at = current[1] if current else now
    rate = rates.get(relation, {}).get("rate_per_day")
    if rate is None:
        return {"still_true": None, "half_life_days": None, "superseded": False, "since": set_at}
    age_days = max(0.0, now - set_at) / DAY
    return {"still_true": round(math.exp(-rate * age_days), 3),
            "half_life_days": round(math.log(2) / rate, 1), "superseded": False, "since": set_at}


def annotate(facts, db_path, user_id, agent_id, now, a0=1.0, b0_days=365.0):
    """Adds freshness to receipt facts ({"fact", "value", ...}) in place."""
    if not facts:
        return
    history = slot_history(db_path, user_id, agent_id)
    rates, runs = fit_rates(history, now, a0, b0_days), _runs(history)
    for f in facts:
        slot_id = f.get("fact") or ""
        relation = slot_id.split("::", 1)[1] if "::" in slot_id else ""
        f["freshness"] = assess(slot_id, relation, f.get("value"), runs, rates, now)
