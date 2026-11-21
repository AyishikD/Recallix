import sqlite3
import time

from backend.utils.paths import get_db_path


class ReflectiveMemory:
    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/sqlite_db/reflections.db")
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                summary TEXT,
                source_memories TEXT,
                confidence REAL,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()

    def store_reflection(self, user_id, summary, source_ids, confidence):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute('''
                INSERT INTO reflections (user_id, summary, source_memories, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, summary, ",".join(map(str, source_ids)), confidence, time.time()))
        conn.close()

    def get_reflections(self, user_id, limit=5):
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute('''
            SELECT summary, confidence, timestamp FROM reflections
            WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()
        return [{"summary": r[0], "confidence": r[1], "timestamp": r[2]} for r in rows]

    def count(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        n = conn.execute("SELECT COUNT(*) FROM reflections WHERE user_id = ?", (user_id,)).fetchone()[0]
        conn.close()
        return n

    def clear_user(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM reflections WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM reflections")
        conn.close()
