"""
Verifiable deletion: receipts and a tamper-evident deletion log.

Deleting a memory removes every derived copy (row, full-text entry, question keys, vector,
long-term copy, timeline entry, confirmed answers, logged ranking features). Afterwards each
store is queried again, and the receipt records what was removed and that nothing is left.

Each receipt is appended to a hash chain:  entry_hash = sha256(prev_hash || canonical receipt).
Changing, inserting or removing any past entry breaks every later hash, which `verify()`
detects (python -m backend.tools.audit verify). The log never stores memory content — only
its SHA-256 — and stores users as SHA-256 hashes of their ids, so it is itself not personal data
beyond what a hash reveals.
"""
import hashlib
import json
import sqlite3
import time

from backend.utils.paths import get_db_path

GENESIS = "0" * 64


def _h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _canonical(receipt):
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AuditLog:
    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/sqlite_db/audit.db")
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS deletions (
                idx INTEGER PRIMARY KEY AUTOINCREMENT, user_hash TEXT, receipt TEXT, prev_hash TEXT, entry_hash TEXT)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_del_user ON deletions(user_hash)")
        conn.close()

    @staticmethod
    def user_hash(user_id):
        return _h(f"recallic-user:{user_id}")

    def record(self, user_id, receipt):
        """Appends `receipt` to the chain; returns it with log_index, prev_hash and entry_hash."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:  # one transaction: read the head and append atomically
                row = conn.execute("SELECT entry_hash FROM deletions ORDER BY idx DESC LIMIT 1").fetchone()
                prev = row[0] if row else GENESIS
                body = {**receipt, "user_hash": self.user_hash(user_id)}
                entry = _h(prev + _canonical(body))
                cur = conn.execute("INSERT INTO deletions (user_hash, receipt, prev_hash, entry_hash) VALUES (?, ?, ?, ?)",
                                   (body["user_hash"], _canonical(body), prev, entry))
                return {**receipt, "log_index": cur.lastrowid, "prev_hash": prev, "entry_hash": entry}
        finally:
            conn.close()

    def receipts(self, user_id, limit=100):
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute("SELECT idx, receipt, prev_hash, entry_hash FROM deletions WHERE user_hash = ? "
                            "ORDER BY idx DESC LIMIT ?", (self.user_hash(user_id), int(limit))).fetchall()
        conn.close()
        return [{**json.loads(r), "log_index": i, "prev_hash": p, "entry_hash": e} for i, r, p, e in rows]

    def verify(self):
        """(ok, entries, first_bad_index or None): recomputes the whole chain."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute("SELECT idx, receipt, prev_hash, entry_hash FROM deletions ORDER BY idx").fetchall()
        conn.close()
        prev = GENESIS
        for idx, receipt, prev_hash, entry_hash in rows:
            if prev_hash != prev or _h(prev + receipt) != entry_hash:
                return False, len(rows), idx
            prev = entry_hash
        return True, len(rows), None


def deletion_receipt(memory_id, content, removed, verified_absent, reason="user request"):
    return {"action": "delete_memory", "reason": reason, "memory_id": int(memory_id),
            "content_sha256": _h(content or ""), "removed": removed, "verified_absent": verified_absent,
            "all_copies_gone": all(verified_absent.values()), "deleted_at": time.time()}
