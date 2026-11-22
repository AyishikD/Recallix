"""
Replay and diff: what Recallix derived from your messages, and what would change.

Every memory keeps its original message, so the derived state (extracted facts,
topics, importance) can be rebuilt by re-running the deterministic store pipeline
over the message log with the original timestamps. Uses:

  * determinism check   replay twice; the snapshots must be identical
  * upgrade preview     replay with new rules or models; diff against today's memory
  * audit               a canonical, human-readable export of what was derived

    python -m backend.tools.replay snapshot --out now.json [--user alice]
    python -m backend.tools.replay replay   --out new.json [--user alice] [--embedder configured|hash|st:...]
    python -m backend.tools.replay diff now.json new.json        # exit 1 if they differ

Replay runs in a sandbox (temporary storage, in-process exact vector engine), never
against production data. Note: the importance score's information-content feature
uses term statistics over all stored memories, so replay every user (the default)
to reproduce importance exactly; a single-user replay reproduces facts and topics
exactly and importance approximately.
"""
import argparse
import hashlib
import json
import sqlite3
import sys

IMPORTANCE_DECIMALS = 3


def _rows(db_path, users=None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql = "SELECT id, user_id, agent_id, memory_type, content, timestamp, importance, metadata FROM episodic_events"
    args = ()
    if users:
        sql += f" WHERE user_id IN ({','.join('?' * len(users))})"
        args = tuple(users)
    rows = [dict(r) for r in conn.execute(sql + " ORDER BY timestamp, id", args)]
    conn.close()
    return rows


def _record(row):
    meta = {}
    try:
        meta = json.loads(row["metadata"] or "{}")
    except ValueError:
        pass
    content = row["content"] or ""
    facts = []
    for slot in meta.get("slots", []) or []:
        src = slot.get("source") or {}
        span = src.get("span")
        facts.append({"fact": slot.get("slot_id"), "value": slot.get("value"), "rule": src.get("rule"),
                      "quote": content[span[0]:span[1]] if span else None})
    return {
        "key": f"{row['user_id']}|{row['timestamp']!r}|{hashlib.sha256(content.encode()).hexdigest()[:16]}",
        "user": row["user_id"], "timestamp": row["timestamp"], "agent": row["agent_id"],
        "memory_type": row["memory_type"], "content": content,
        "speaker": (meta.get("client") or {}).get("speaker"),
        "facts": sorted(facts, key=lambda f: (f["fact"] or "", f["value"] or "")),
        "schemas": sorted(meta.get("schema_tags") or []),
        "importance": round(float(row["importance"] or 0.0), IMPORTANCE_DECIMALS),
    }


def snapshot(db_path, users=None):
    from backend.brain.features.facts import EXTRACTOR_VERSION
    return {"extractor": EXTRACTOR_VERSION, "memories": [_record(r) for r in _rows(db_path, users)]}


def production_db():
    from backend.brain.memory.episodic.episodic_memory import EpisodicMemory
    return EpisodicMemory().db_path


def replay(source_rows, embedder="configured"):
    """Re-run the store pipeline over `source_rows` (in timestamp order) in a sandbox."""
    from backend.tools.sandbox import Sandbox
    with Sandbox(embedder=embedder) as box:
        with box.silenced():
            for r in sorted(source_rows, key=lambda r: (r["timestamp"], r["id"])):
                meta = {}
                try:
                    meta = json.loads(r["metadata"] or "{}")
                except ValueError:
                    pass
                box.mm.store(r["user_id"], r["content"], agent_id=r["agent_id"] or "default_agent",
                             memory_type=r["memory_type"] or "private", client_metadata=meta.get("client"),
                             timestamp=r["timestamp"])
        return snapshot(box.mm.episodic.db_path)


def diff(a, b):
    """Differences between two snapshots, as a list of human-readable lines."""
    left = {m["key"]: m for m in a["memories"]}
    right = {m["key"]: m for m in b["memories"]}
    lines = []
    for key in sorted(set(left) - set(right)):
        lines.append(f"- only in first:  {left[key]['user']}: {left[key]['content'][:80]!r}")
    for key in sorted(set(right) - set(left)):
        lines.append(f"+ only in second: {right[key]['user']}: {right[key]['content'][:80]!r}")
    for key in sorted(set(left) & set(right)):
        x, y = left[key], right[key]
        head = f"~ {x['user']}: {x['content'][:70]!r}"
        fx = {(f["fact"], f["value"]) for f in x["facts"]}
        fy = {(f["fact"], f["value"]) for f in y["facts"]}
        changes = [f"fact removed {f}={v!r}" for f, v in sorted(fx - fy)] + \
                  [f"fact added {f}={v!r}" for f, v in sorted(fy - fx)]
        if x["schemas"] != y["schemas"]:
            changes.append(f"topics {x['schemas']} -> {y['schemas']}")
        if abs(x["importance"] - y["importance"]) > 10 ** -IMPORTANCE_DECIMALS:
            changes.append(f"importance {x['importance']} -> {y['importance']}")
        if changes:
            lines.append(head)
            lines.extend("    " + c for c in changes)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="export what is currently derived")
    s.add_argument("--out", required=True)
    s.add_argument("--user", action="append")
    r = sub.add_parser("replay", help="re-derive from the original messages in a sandbox")
    r.add_argument("--out", required=True)
    r.add_argument("--user", action="append")
    r.add_argument("--embedder", default="configured")
    d = sub.add_parser("diff", help="compare two snapshots (exit 1 if they differ)")
    d.add_argument("first")
    d.add_argument("second")
    args = ap.parse_args(argv)

    if args.cmd == "snapshot":
        snap = snapshot(production_db(), args.user)
    elif args.cmd == "replay":
        snap = replay(_rows(production_db(), args.user), args.embedder)
    else:
        a, b = (json.load(open(p)) for p in (args.first, args.second))
        lines = diff(a, b)
        if a.get("extractor") != b.get("extractor"):
            print(f"extractor rules: {a.get('extractor')} -> {b.get('extractor')}")
        print("\n".join(lines) if lines else "identical")
        return 1 if lines else 0
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: {len(snap['memories'])} memories (extractor {snap['extractor']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
