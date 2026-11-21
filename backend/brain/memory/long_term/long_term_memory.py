import sqlite3
import time

from backend.utils.paths import get_db_path


class LongTermMemory:
    """High-importance facts promoted from episodic memory (one row per source memory)."""

    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/sqlite_db/memory.db")
        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS long_term_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                fact TEXT,
                importance INTEGER,
                access_count INTEGER DEFAULT 1,
                last_updated REAL,
                last_accessed REAL
            )
        ''')
        columns = {r[1] for r in conn.execute("PRAGMA table_info(long_term_facts)")}
        if "source_id" not in columns:
            conn.execute("ALTER TABLE long_term_facts ADD COLUMN source_id INTEGER")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ltf_source ON long_term_facts(user_id, source_id)")
        conn.commit()
        conn.close()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def promote(self, user_id, event):
        conn = self._connect()
        with conn:
            conn.execute('''
                INSERT OR IGNORE INTO long_term_facts (user_id, fact, importance, last_updated, last_accessed, source_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, event["content"], event["importance"], event["timestamp"], event["timestamp"], event.get("id")))
        conn.close()

    def get(self, user_id):
        from ...utils.decay_logic import DecayLogic
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, fact, importance, access_count, last_accessed, source_id FROM long_term_facts WHERE user_id = ?",
                (user_id,)).fetchall()
            results = []
            with conn:
                for fid, fact, imp, count, last, source_id in rows:
                    strength = DecayLogic.calculate_strength(imp, last, count)
                    if strength > 1.0:  # retention threshold
                        results.append({"id": fid, "source_id": source_id, "fact": fact,
                                        "importance": imp, "strength": strength})
                        conn.execute("UPDATE long_term_facts SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                                     (time.time(), fid))
        finally:
            conn.close()
        return sorted(results, key=lambda x: x["strength"], reverse=True)

    def delete_by_source(self, user_id, memory_id):
        conn = self._connect()
        with conn:
            conn.execute("DELETE FROM long_term_facts WHERE user_id = ? AND source_id = ?", (user_id, memory_id))
        conn.close()

    def clear_user(self, user_id):
        conn = self._connect()
        with conn:
            conn.execute("DELETE FROM long_term_facts WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = self._connect()
        with conn:
            conn.execute("DELETE FROM long_term_facts")
        conn.close()
