import json
import os
import sqlite3
import time
from collections import defaultdict, Counter

from backend.brain.features.schemas import SEEDS
from backend.utils.paths import get_db_path


class PredictionEngine:
    """
    Next-topic prediction without an LLM: a first-order Markov chain over the
    user's sequence of memory schemas.

        P(next = b | last = a) = (count(a -> b) + 1) / (count(a -> *) + K)

    (Laplace smoothing over the K schemas.) The prediction is the
    distribution for the user's most recent schema, top 3.
    """

    LABELS = list(SEEDS.keys())

    def __init__(self, storage_path=None, memory_db=None):
        self.storage_path = storage_path or get_db_path("backend/storage/predictive_cache/predictions.json")
        self.memory_db = memory_db or get_db_path("backend/storage/sqlite_db/memory.db")
        self._init_storage()

    def _init_storage(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                json.dump({}, f)

    def schema_sequence(self, user_id, limit=500):
        try:
            conn = sqlite3.connect(self.memory_db)
            rows = conn.execute(
                "SELECT metadata FROM episodic_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)).fetchall()
            conn.close()
        except sqlite3.Error:
            return []
        seq = []
        for (metadata,) in reversed(rows):
            try:
                tags = json.loads(metadata or "{}").get("schema_tags") or ["general"]
            except ValueError:
                tags = ["general"]
            seq.append(tags[0])
        return seq

    @classmethod
    def transition_probabilities(cls, sequence, last):
        counts = defaultdict(Counter)
        for a, b in zip(sequence, sequence[1:]):
            counts[a][b] += 1
        row = counts[last]
        total = sum(row.values())
        k = len(cls.LABELS)
        return {b: (row[b] + 1) / (total + k) for b in cls.LABELS}

    def predict(self, user_id, context_data=None):
        """context_data may carry an explicit 'schema_sequence'; otherwise it is read from memory."""
        seq = (context_data or {}).get("schema_sequence") or self.schema_sequence(user_id)
        if not seq:
            prediction = {"next_topics": [], "based_on": 0}
        else:
            probs = self.transition_probabilities(seq, seq[-1])
            top = sorted(probs.items(), key=lambda x: -x[1])[:3]
            prediction = {"last_topic": seq[-1],
                          "next_topics": [{"topic": t, "probability": round(p, 3)} for t, p in top],
                          "based_on": len(seq)}
        self._save_prediction(user_id, prediction)
        return prediction

    def _save_prediction(self, user_id, prediction):
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        data[user_id] = {"prediction": prediction, "timestamp": time.time()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)

    def get_last_prediction(self, user_id):
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        return data.get(user_id, {}).get("prediction", "No prediction available.")
