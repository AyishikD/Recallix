import json
import os
import sqlite3
import time
from collections import Counter

from backend.utils.paths import get_db_path


class StateInference:
    """
    User world-state without an LLM.

    States come from two deterministic sources over the user's recent memories:
      * explicit facts: the newest value of every user::<relation> fact slot
        (e.g. location, employer, allergy); confidence 0.9.
      * interests: the share of memories per schema, stored as
        interest_<schema>; confidence = that share.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path("backend/storage/world_state_store/states.db")
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                state_key TEXT,
                state_value TEXT,
                confidence REAL,
                last_updated REAL,
                UNIQUE(user_id, state_key)
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def derive_states(recent_memories):
        """recent_memories: dicts with 'content', 'timestamp', 'metadata' (dict or JSON)."""
        states = {}
        schema_counts = Counter()
        for m in sorted(recent_memories, key=lambda x: float(x.get("timestamp", 0))):
            metadata = m.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except ValueError:
                    metadata = {}
            for slot in metadata.get("slots", []):
                if slot.get("entity") == "user":
                    states[slot["relation"]] = (slot.get("value", ""), 0.9)  # newest wins
            for tag in metadata.get("schema_tags", []):
                if tag != "general":
                    schema_counts[tag] += 1
        total = sum(schema_counts.values())
        for tag, count in schema_counts.items():
            share = count / total
            states[f"interest_{tag}"] = (f"{share:.0%}", round(share, 3))
        return states

    def infer_and_update(self, user_id, recent_memories):
        if not recent_memories:
            return {}
        states = self.derive_states(recent_memories)
        conn = sqlite3.connect(self.db_path)
        for key, (value, confidence) in states.items():
            conn.execute('''
                INSERT INTO user_states (user_id, state_key, state_value, confidence, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    confidence = excluded.confidence,
                    last_updated = excluded.last_updated
            ''', (user_id, key, value, confidence, time.time()))
        conn.commit()
        conn.close()
        return states

    def clear_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute("DELETE FROM user_states WHERE user_id = ?", (user_id,))
        conn.close()

    def clear_all(self):
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute("DELETE FROM user_states")
        conn.close()

    def get_states(self, user_id):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute('SELECT state_key, state_value, confidence FROM user_states WHERE user_id = ?', (user_id,)).fetchall()
        conn.close()
        return {row[0]: {"value": row[1], "confidence": row[2]} for row in rows}
