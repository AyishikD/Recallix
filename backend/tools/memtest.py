"""
Memory tests: assert what your agent remembers — and refuses to invent — in CI.

Store pipeline and recall are deterministic (no LLM), so a memory test gives the same
result on every run: a failing test means behaviour changed, not that a model was in
a different mood. Each spec runs in its own sandbox (temporary storage, in-process
exact vector engine); production data is never touched.

    python -m backend.tools.memtest tests/memory/*.yaml [--embedder hash] [--answer-check on|off]

Spec (YAML; one file may hold a list of specs):

    name: remembers allergies, not other people's
    store:
      - "I'm allergic to peanuts and prefer dark chocolate."
      - {text: "Bob mentioned he loves hiking.", agent: support, type: shared, speaker: Priya}
    expect:
      - recall: "What am I allergic to?"
        status: CONSISTENT            # CONSISTENT | ABSTAINED | NO_MATCH
        top_contains: peanuts         # the first returned memory contains this text
      - recall: "Which sport does Alice play?"
        status: [ABSTAINED, NO_MATCH] # any of these
      - fact: user::allergy           # an extracted fact (slot id) ...
        contains: peanuts             # ... whose value contains this text
      - recall: "When does Atlas launch?"
        answer_contains: "July 14"    # needs the answer check (--answer-check on)

Exit code 0 when every expectation holds, 1 otherwise.
"""
import argparse
import glob
import json
import sys

import yaml


def _load(paths):
    specs = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            data = yaml.safe_load(open(path))
            for i, spec in enumerate(data if isinstance(data, list) else [data]):
                spec.setdefault("name", f"{path}#{i + 1}")
                specs.append(spec)
    return specs


def _all_facts(db_path, user):
    import sqlite3
    conn = sqlite3.connect(db_path)
    out = {}
    for (meta,) in conn.execute("SELECT metadata FROM episodic_events WHERE user_id = ? ORDER BY timestamp, id", (user,)):
        try:
            for s in (json.loads(meta or "{}").get("slots") or []):
                out.setdefault(s["slot_id"], []).append(s.get("value", ""))
        except ValueError:
            continue
    conn.close()
    return out


def run_spec(spec, embedder, answer_check):
    from backend.tools.sandbox import Sandbox
    user = spec.get("user", "memtest-user")
    results = []
    with Sandbox(embedder=embedder, answer_check=answer_check) as box:
        with box.silenced():
            for i, item in enumerate(spec.get("store", [])):
                item = {"text": item} if isinstance(item, str) else item
                client = {"speaker": item["speaker"]} if item.get("speaker") else None
                box.mm.store(user, item["text"], agent_id=item.get("agent", "default_agent"),
                             memory_type=item.get("type", "private"), client_metadata=client,
                             timestamp=item.get("timestamp", 1_700_000_000 + i))  # fixed clock: reproducible
        facts = _all_facts(box.mm.episodic.db_path, user)
        for exp in spec.get("expect", []):
            if "fact" in exp:
                values = facts.get(exp["fact"], [])
                want = exp.get("contains", exp.get("equals"))
                ok = bool(values) if want is None else any(
                    (v == want) if "equals" in exp else (str(want).lower() in str(v).lower()) for v in values)
                results.append((ok, f"fact {exp['fact']}" + (f" ~ {want!r}" if want else ""), f"got {values or 'nothing'}"))
                continue
            with box.silenced():
                out = box.mm.retrieve(user, exp["recall"], agent_id=exp.get("agent", "default_agent"),
                                      limit=exp.get("limit", 5))
            mems = out["memories"]
            got = f"status {out['status']}, top {mems[0]['content'][:60]!r}" if mems else f"status {out['status']}, nothing returned"
            checks = []
            if "status" in exp:
                allowed = exp["status"] if isinstance(exp["status"], list) else [exp["status"]]
                checks.append(out["status"] in allowed)
            if "top_contains" in exp:
                checks.append(bool(mems) and exp["top_contains"].lower() in mems[0]["content"].lower())
            if "contains" in exp:
                checks.append(any(exp["contains"].lower() in m["content"].lower() for m in mems))
            if "not_contains" in exp:
                checks.append(not any(exp["not_contains"].lower() in m["content"].lower() for m in mems))
            if "answer_contains" in exp:
                answers = [m.get("answer") or "" for m in mems]
                checks.append(any(exp["answer_contains"].lower() in a.lower() for a in answers))
                got += f", answers {[a for a in answers if a]}"
            label = f"recall {exp['recall']!r}"
            results.append((all(checks) if checks else True, label, got + f" — {out['receipt']['reason']}"))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="YAML spec files or globs")
    ap.add_argument("--embedder", default="configured", help="configured | hash | st:<model> (hash: fast, for CI)")
    ap.add_argument("--answer-check", choices=["on", "off", "config"], default="config")
    args = ap.parse_args(argv)
    answer_check = {"on": True, "off": False, "config": None}[args.answer_check]

    failures = total = 0
    for spec in _load(args.specs):
        print(spec["name"])
        for ok, label, detail in run_spec(spec, args.embedder, answer_check):
            total += 1
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"\n        {detail}"))
    print(f"\n{total - failures}/{total} expectations passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
