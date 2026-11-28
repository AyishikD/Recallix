"""
Memory fuzzer: generate memory tests automatically from what Recallic extracted.

    python -m backend.tools.fuzz --spec tests/memory/example.yaml [--write-failures fails.yaml]
    python -m backend.tools.fuzz --eval dev                 # the dev scenarios (eval/data/dev/retrieval.jsonl)
    python -m backend.tools.fuzz --user alice               # a real user's messages, replayed in a sandbox

For every extracted fact (e.g. user::location = Pune) three kinds of case are generated:

  ask      the questions the fact answers, phrased several ways  -> the top memory must hold the value
  someone  the same question about a person who is not in memory -> must NOT return this fact
  change   store an update with a new value, then ask             -> the new value must win, and the
                                                                     store response must flag a conflict

The store pipeline is deterministic, so every failure reproduces exactly; failures are written
as ready-to-run memtest YAML. Each recall case also records the signals of the top memory
(relevance, answer margin, subject contrast, freshness) with whether it was right — the
labelled data that fits the per-answer trust score (backend.tools.fit_trust).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTSIDERS = ["Morgan", "Quinn", "Rowan", "Ellis", "Harper", "Sasha"]
NEW_VALUES = {
    "location": ["Lisbon", "Nairobi", "Hanoi", "Montreal", "Valparaiso"],
    "name": ["Devika", "Tomasz", "Ifeoma", "Mateo", "Aiko"],
    "deadline": ["March 9", "October 21", "June 30"], "when": ["March 9", "October 21", "June 30"],
    "birthday": ["2 February", "19 November"], "launch": ["March 9", "October 21"],
    "job": ["nurse", "architect", "pilot"], "employer": ["Infosys", "Siemens", "Canva"],
}
GENERIC_VALUES = ["Zephyr", "Kestrel", "Orion"]


def _poss(entity, display):
    return "my" if entity == "user" else f"{display}'s"


def ask_questions(slot, display):
    from backend.brain.features.facts import question_keys
    qs = question_keys(slot, display_entity=display)  # natural phrasings per relation (generic one only as fallback)
    return list(dict.fromkeys(qs))[:3]


def someone_question(slot):
    """The same question about an outsider (only for facts about the user)."""
    from backend.brain.features.facts import question_keys
    if slot["entity"] != "user":
        return None
    name = OUTSIDERS[sum(map(ord, slot["slot_id"])) % len(OUTSIDERS)]
    qs = question_keys({**slot, "entity": name.lower()}, display_entity=name)
    return (qs[0] if qs else f"What is {name}'s {slot['relation'].replace('_', ' ')}?"), name


def update_message(slot, display, rng):
    """A message that changes the fact to a new value the rule grammar extracts, or None."""
    rel, ent = slot["relation"], slot["entity"]
    pool = [v for v in NEW_VALUES.get(rel, GENERIC_VALUES) if v.lower() not in str(slot["value"]).lower()]
    new = rng.choice(pool)
    if ent == "user":
        if rel == "location":
            return f"I moved to {new}.", new
        return f"My {rel.replace('_', ' ')} is now {new}.", new
    if rel == "when":
        return f"The {display} was moved to {new}.", new
    return f"The {display} {rel.replace('_', ' ')} is now {new}.", new


def _facts(db_path, user):
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT e.id, e.content, j.value FROM episodic_events e, json_each(e.metadata, '$.slots') j "
                        "WHERE e.user_id = ? ORDER BY e.timestamp, e.id", (user,)).fetchall()
    conn.close()
    out = {}
    texts = list(dict.fromkeys(content for _, content, _ in rows))
    for mid, content, slot in rows:
        s = json.loads(slot)
        if s.get("relation") in ("alias_of",):
            continue
        out[s["slot_id"]] = {**s, "memory_id": mid, "content": content, "display": None}
    for s in out.values():  # the entity as the user wrote it ("Project Alpha" for projectalpha), from any
        cand = s["entity"].replace("_", " ")  # message: a later "the deadline was moved" may not name it
        if s["entity"] != "user":
            found = None
            for text in [s["content"]] + texts:
                for m in re.finditer(r"([A-Z][\w-]*(?:\s+[A-Z0-9][\w-]*)*)", text):
                    span = re.sub(r"^(?:The|A|An)\s+", "", m.group(1))
                    if re.sub(r"[^a-z0-9]", "", span.lower()) == s["entity"]:
                        found = span
                        break
                if found:
                    break
            cand = found or cand
        s["display"] = cand
    return out  # latest value per slot


def _signals(memory):
    r = memory.get("receipt") or {}
    fresh = next((f.get("freshness", {}).get("still_true") for f in r.get("facts", []) if f.get("freshness")), None)
    return {"relevance": memory.get("relevance"), "answer_margin": memory.get("answer_margin"),
            "subject_contrast": memory.get("subject_contrast"), "still_true": fresh}


def fuzz(messages, embedder="hash", answer_check=False, seed=0):
    """messages: [{"text", "speaker"?, "agent"?, "type"?}] -> {"cases": [...], "summary": {...}}."""
    from backend.tools.sandbox import Sandbox
    rng = random.Random(seed)
    user = "fuzz-user"

    def seed_box(box):
        with box.silenced():
            for i, m in enumerate(messages):
                box.mm.store(user, m["text"], agent_id=m.get("agent", "default_agent"),
                             memory_type=m.get("type", "private"),
                             client_metadata={"speaker": m["speaker"]} if m.get("speaker") else None,
                             timestamp=1_700_000_000 + i)

    def recall(box, q):
        with box.silenced():
            return box.mm.retrieve(user, q, limit=5)

    cases = []
    with Sandbox(embedder=embedder, answer_check=answer_check) as box:
        seed_box(box)
        facts = _facts(box.mm.episodic.db_path, user)
        for slot in facts.values():
            value = str(slot["value"]).lower()
            for q in ask_questions(slot, slot["display"]):
                out = recall(box, q)
                top = out["memories"][0] if out["memories"] else None
                ok = bool(top) and value in top["content"].lower()
                cases.append({"kind": "ask", "fact": slot["slot_id"], "question": q, "expect": slot["value"],
                              "passed": ok, "status": out["status"], "top": top["content"] if top else None,
                              "signals": _signals(top) if top else None, "top_correct": ok if top else None})
            sq = someone_question(slot)
            if sq:
                q, name = sq
                out = recall(box, q)
                top = out["memories"][0] if out["memories"] else None
                leaked = bool(top) and value in top["content"].lower()
                cases.append({"kind": "someone", "fact": slot["slot_id"], "question": q, "outsider": name,
                              "must_not_return": slot["value"], "passed": not leaked, "status": out["status"],
                              "top": top["content"] if top else None,
                              "signals": _signals(top) if top else None, "top_correct": False if top else None})
    for slot in facts.values():
        msg, new = update_message(slot, slot["display"], rng)
        with Sandbox(embedder=embedder, answer_check=answer_check) as box:
            seed_box(box)
            with box.silenced():
                res = box.mm.store(user, msg, timestamp=1_700_000_000 + len(messages) + 1)
            flagged = any(c["fact"] == slot["slot_id"] for c in res.get("conflicts", []))
            q = ask_questions(slot, slot["display"])[0]
            out = recall(box, q)
            top = out["memories"][0] if out["memories"] else None
            ok = bool(top) and new.lower() in top["content"].lower()
            cases.append({"kind": "change", "fact": slot["slot_id"], "update": msg, "question": q, "expect": new,
                          "passed": ok and flagged, "new_value_wins": ok, "conflict_flagged": flagged,
                          "top": top["content"] if top else None})
    summary = {}
    for kind in ("ask", "someone", "change"):
        ks = [c for c in cases if c["kind"] == kind]
        summary[kind] = {"cases": len(ks), "passed": sum(c["passed"] for c in ks),
                         "pass_rate": round(sum(c["passed"] for c in ks) / len(ks), 3) if ks else None}
    return {"facts": len(facts), "cases": cases, "summary": summary}


def failures_as_memtests(messages, report):
    """Failing cases as memtest specs (reproducible: same messages, same pipeline)."""
    store = [{"text": m["text"], **({"speaker": m["speaker"]} if m.get("speaker") else {})} for m in messages]
    specs = []
    for c in report["cases"]:
        if c["passed"]:
            continue
        if c["kind"] == "ask":
            specs.append({"name": f"fuzz ask: {c['question']}", "store": store,
                          "expect": [{"recall": c["question"], "top_contains": c["expect"]}]})
        elif c["kind"] == "someone":
            specs.append({"name": f"fuzz someone else: {c['question']}", "store": store,
                          "expect": [{"recall": c["question"], "not_contains": c["must_not_return"]}]})
        else:
            specs.append({"name": f"fuzz change: {c['update']}", "store": store + [{"text": c["update"]}],
                          "expect": [{"recall": c["question"], "top_contains": c["expect"]}]})
    return specs


def _load(args):
    if args.spec:
        import yaml
        data = yaml.safe_load(open(args.spec))
        msgs = []
        for spec in data if isinstance(data, list) else [data]:
            msgs += [{"text": s} if isinstance(s, str) else s for s in spec.get("store", [])]
        return [msgs]
    if args.eval:
        rows = [json.loads(l) for l in open(REPO / "eval" / "data" / args.eval / "retrieval.jsonl") if l.strip()]
        return [[{"text": m} for m in sc["memories"]] for sc in rows]
    import sqlite3
    from backend.tools.replay import production_db
    conn = sqlite3.connect(production_db())
    rows = conn.execute("SELECT content, agent_id, memory_type, metadata FROM episodic_events WHERE user_id = ? "
                        "ORDER BY timestamp, id", (args.user,)).fetchall()
    conn.close()
    return [[{"text": c, "agent": a, "type": t,
              "speaker": (json.loads(md or "{}").get("client") or {}).get("speaker")} for c, a, t, md in rows]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec")
    src.add_argument("--eval", choices=["dev"])
    src.add_argument("--user")
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--answer-check", choices=["on", "off"], default="off")
    ap.add_argument("--json", help="write the full report (cases with signals) here")
    ap.add_argument("--write-failures", help="write failing cases as memtest YAML here")
    args = ap.parse_args(argv)

    reports, specs = [], []
    for messages in _load(args):
        rep = fuzz(messages, args.embedder, args.answer_check == "on")
        reports.append(rep)
        specs += failures_as_memtests(messages, rep)
    cases = [c for r in reports for c in r["cases"]]
    print(f"{sum(r['facts'] for r in reports)} facts -> {len(cases)} generated cases")
    for kind in ("ask", "someone", "change"):
        ks = [c for c in cases if c["kind"] == kind]
        if ks:
            print(f"  {kind:8s} {sum(c['passed'] for c in ks)}/{len(ks)} passed")
    for c in [c for c in cases if not c["passed"]][:15]:
        print(f"  FAIL {c['kind']:8s} {c['question']!r} -> {(c.get('top') or 'nothing')[:70]!r}")
    if args.json:
        Path(args.json).write_text(json.dumps({"reports": reports}, indent=1, default=str))
    if args.write_failures and specs:
        import yaml
        Path(args.write_failures).write_text(yaml.safe_dump(specs, sort_keys=False, allow_unicode=True))
        print(f"wrote {len(specs)} failing cases as memtests to {args.write_failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
