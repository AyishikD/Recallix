"""
Portable memory: export a user's memories in an open format, and import from Recallix,
Mem0 or plain text. Memory belongs to the user, not to a model vendor or to us.

    python -m backend.tools.portable export --user alice --out alice.json
    python -m backend.tools.portable import --user alice --file alice.json            # Recallix export
    python -m backend.tools.portable import --user alice --file mem0.json --format mem0
    python -m backend.tools.portable import --user alice --file notes.txt --format text  # one memory per line

Format "recallix-memory" v1 (JSON):

    {"format": "recallix-memory", "version": 1, "exported_at": "...", "user_id": "...",
     "extractor": "rules-...", "embedding_model": "...",
     "memories": [{"content": "...", "timestamp": 1700000000.0, "agent_id": "...",
                   "memory_type": "private|shared", "metadata": {...client metadata...},
                   "derived": {"facts": [...], "topics": [...], "importance": 6.2}}]}

The original messages are the source of truth. `derived` is informational: import re-runs
the deterministic store pipeline with the original timestamps, so a file moves cleanly
between Recallix versions and embedding models (the result is what `replay` would give).
"""
import argparse
import datetime as _dt
import json
import sqlite3
import sys
import time

FORMAT, VERSION = "recallix-memory", 1
MAX_IMPORT_ITEMS = 5000


def export_user(mm, user_id):
    from backend.brain.features.facts import EXTRACTOR_VERSION
    conn = sqlite3.connect(mm.episodic.db_path)
    rows = conn.execute("SELECT content, timestamp, agent_id, memory_type, importance, metadata FROM episodic_events "
                        "WHERE user_id = ? ORDER BY timestamp, id", (user_id,)).fetchall()
    conn.close()
    memories = []
    for content, ts, agent_id, memory_type, importance, metadata in rows:
        try:
            meta = json.loads(metadata or "{}")
        except ValueError:
            meta = {}
        memories.append({
            "content": content, "timestamp": ts, "agent_id": agent_id, "memory_type": memory_type,
            "metadata": meta.get("client") or {},
            "derived": {"facts": [{"fact": s.get("slot_id"), "value": s.get("value")} for s in meta.get("slots") or []],
                        "topics": meta.get("schema_tags") or [], "importance": importance},
        })
    return {"format": FORMAT, "version": VERSION,
            "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "user_id": user_id, "extractor": EXTRACTOR_VERSION, "embedding_model": mm.semantic.model_name,
            "memories": memories}


def _timestamp(value, fallback):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return fallback


def parse(data, fmt="recallix"):
    """Normalise an export into [{"content", "timestamp", "agent_id", "memory_type", "metadata"}]."""
    now = time.time()
    if fmt == "recallix":
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(f'not a "{FORMAT}" export')
        if int(data.get("version", 0)) > VERSION:
            raise ValueError(f"export version {data.get('version')} is newer than this Recallix ({VERSION})")
        items = [{"content": m.get("content", ""), "timestamp": _timestamp(m.get("timestamp"), now),
                  "agent_id": m.get("agent_id"), "memory_type": m.get("memory_type"),
                  "metadata": m.get("metadata") or None} for m in data.get("memories", [])]
    elif fmt == "mem0":
        rows = data.get("results", data.get("memories", [])) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("expected a list of Mem0 memories (or {\"results\": [...]})")
        items = []
        for i, m in enumerate(rows):
            text = m.get("memory") or m.get("text") or m.get("content") if isinstance(m, dict) else str(m)
            items.append({"content": text or "", "timestamp": _timestamp(m.get("created_at") if isinstance(m, dict) else None, now + i * 1e-3),
                          "metadata": (m.get("metadata") or None) if isinstance(m, dict) else None})
    elif fmt == "text":
        lines = data.splitlines() if isinstance(data, str) else list(data)
        items = [{"content": str(line).strip().lstrip("-•* ").strip(), "timestamp": now + i * 1e-3}
                 for i, line in enumerate(lines)]
    else:
        raise ValueError(f"unknown format {fmt!r} (recallix, mem0, text)")
    items = [it for it in items if it["content"].strip()]
    if len(items) > MAX_IMPORT_ITEMS:
        raise ValueError(f"{len(items)} memories in one import; the limit is {MAX_IMPORT_ITEMS} per request")
    return items


def import_items(mm, user_id, items, agent_id="default_agent", memory_type="private"):
    """Stores items in time order through the normal pipeline (duplicates are skipped)."""
    stored = skipped = 0
    for it in sorted(items, key=lambda x: x["timestamp"]):
        mt = it.get("memory_type") if it.get("memory_type") in ("private", "shared") else memory_type
        res = mm.store(user_id, it["content"], agent_id=it.get("agent_id") or agent_id, memory_type=mt,
                       client_metadata=it.get("metadata"), timestamp=it["timestamp"])
        stored += res["status"] == "stored"
        skipped += res["status"] != "stored"
    return {"stored": stored, "skipped_duplicates": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--user", required=True)
    e.add_argument("--out", required=True)
    i = sub.add_parser("import")
    i.add_argument("--user", required=True)
    i.add_argument("--file", required=True)
    i.add_argument("--format", default="recallix", choices=["recallix", "mem0", "text"])
    i.add_argument("--agent", default="default_agent")
    i.add_argument("--memory-type", default="private", choices=["private", "shared"])
    args = ap.parse_args(argv)

    from backend.api.deps import get_memory_manager
    mm = get_memory_manager()
    if args.cmd == "export":
        data = export_user(mm, args.user)
        with open(args.out, "w") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"exported {len(data['memories'])} memories of {args.user} to {args.out}")
        return 0
    raw = open(args.file).read()
    data = raw if args.format == "text" else json.loads(raw)
    print(json.dumps(import_items(mm, args.user, parse(data, args.format), args.agent, args.memory_type)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
