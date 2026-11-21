import json
import sqlite3
import time

from backend.utils.paths import get_db_path


class ArchiveStore:
    """Memories moved out of active recall by the forgetting policy (kept, not searched)."""

    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/archive_store/archived_memories.db")
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS archived_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                timestamp REAL,
                importance REAL,
                reinforcement_score REAL,
                schema_tags TEXT,
                archived_at REAL
            )
        ''')
        conn.commit()
        conn.close()

    def archive(self, memory):
        metadata = memory.get("metadata") or "{}"
        try:
            tags = json.loads(metadata).get("schema_tags", []) if isinstance(metadata, str) else metadata.get("schema_tags", [])
        except ValueError:
            tags = []
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute('''
                INSERT OR REPLACE INTO archived_memories
                (id, user_id, content, timestamp, importance, reinforcement_score, schema_tags, archived_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(memory.get("id")), memory.get("user_id"), memory.get("content"), memory.get("timestamp"),
                  memory.get("importance"), memory.get("reinforcement_score"), json.dumps(tags), time.time()))
        conn.close()

    def count(self, user_id=None):
        conn = sqlite3.connect(self.db_path, timeout=30)
        q, args = ("SELECT COUNT(*) FROM archived_memories", ()) if user_id is None else \
            ("SELECT COUNT(*) FROM archived_memories WHERE user_id = ?", (user_id,))
        n = conn.execute(q, args).fetchone()[0]
        conn.close()
        return n

    def clear_user(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM archived_memories WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM archived_memories")
        conn.close()
