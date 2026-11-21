import sqlite3
import time

from backend.utils.paths import get_db_path


class TimelineEngine:
    """Chronological log of each user's stored memories (written on every store)."""

    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/timeline_store/timeline.db")
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                event_content TEXT,
                timestamp REAL
            )
        ''')
        columns = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        if "memory_id" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN memory_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, timestamp)")
        conn.commit()
        conn.close()

    def append_event(self, user_id, content, memory_id=None, timestamp=None):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("INSERT INTO events (user_id, event_content, timestamp, memory_id) VALUES (?, ?, ?, ?)",
                         (user_id, content, timestamp or time.time(), memory_id))
        conn.close()

    def get_sequence(self, user_id, limit=50):
        """The user's most recent `limit` events, oldest first."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute('''
            SELECT event_content, timestamp, memory_id FROM events
            WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (user_id, limit)).fetchall()
        conn.close()
        return [{"content": r[0], "timestamp": r[1], "memory_id": r[2]} for r in reversed(rows)]

    def count(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        n = conn.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", (user_id,)).fetchone()[0]
        conn.close()
        return n

    def delete_by_memory(self, user_id, memory_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM events WHERE user_id = ? AND memory_id = ?", (user_id, memory_id))
        conn.close()

    def clear_user(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM events WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM events")
        conn.close()
