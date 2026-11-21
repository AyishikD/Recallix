import json
import os
import sqlite3
import time

from backend import settings
from backend.utils.paths import get_db_path


class WorkingMemory:
    """
    Sliding window of each user's most recent messages, one SQLite row per
    user (a write touches only that user's row; safe under concurrent requests).
    A legacy working_memory.json is imported once and renamed.
    """

    def __init__(self, db_path=None, window=None):
        self.db_path = db_path or get_db_path("backend/storage/sqlite_db/working_memory.db")
        self.window = int(window if window is not None else settings.get("memory.working_window", 15))
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS working_memory (user_id TEXT PRIMARY KEY, history TEXT, updated REAL)")
        conn.commit()
        conn.close()
        self._import_legacy_json()

    def _import_legacy_json(self):
        legacy = get_db_path("backend/storage/working_memory.json")
        if not os.path.exists(legacy):
            return
        try:
            with open(legacy) as f:
                data = json.load(f)
            conn = sqlite3.connect(self.db_path, timeout=30)
            for user_id, entry in data.items():
                conn.execute("INSERT OR IGNORE INTO working_memory VALUES (?, ?, ?)",
                             (user_id, json.dumps(entry.get("history", [])[-self.window:]), time.time()))
            conn.commit()
            conn.close()
            os.replace(legacy, legacy + ".imported")
        except (OSError, ValueError) as e:
            print(f"[WorkingMemory] could not import legacy JSON: {e}")

    def update(self, user_id, message):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:  # one transaction: read-modify-write of this user's row
                row = conn.execute("SELECT history FROM working_memory WHERE user_id = ?", (user_id,)).fetchone()
                history = (json.loads(row[0]) if row else []) + [message]
                conn.execute("INSERT OR REPLACE INTO working_memory VALUES (?, ?, ?)",
                             (user_id, json.dumps(history[-self.window:]), time.time()))
        finally:
            conn.close()

    def get(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            row = conn.execute("SELECT history FROM working_memory WHERE user_id = ?", (user_id,)).fetchone()
        finally:
            conn.close()
        history = json.loads(row[0]) if row else []
        return {"history": history, "summary": " | ".join(history)}

    def clear_user(self, user_id):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM working_memory WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute("DELETE FROM working_memory")
        conn.close()
