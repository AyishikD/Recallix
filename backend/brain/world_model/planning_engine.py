import json
import math
import sqlite3
import time


class PlanningEngine:
    """
    Proactive plan without an LLM: the user's open task and calendar memories,
    ordered by priority

        priority = (importance / 10) * exp(-age_days / 14) * (1 + 0.5 * predicted_topic_match)

    so important, recent items come first, nudged towards the topics the
    PredictionEngine expects next.
    """

    PLAN_SCHEMAS = {"task", "calendar"}

    def __init__(self, memory_manager):
        self.memory = memory_manager

    def generate_plan(self, user_id, prediction=None, current_state=None, max_steps=5):
        predicted = {t["topic"] for t in (prediction or {}).get("next_topics", [])} \
            if isinstance(prediction, dict) else set()
        conn = sqlite3.connect(self.memory.episodic.db_path)
        rows = conn.execute(
            "SELECT id, content, importance, timestamp, metadata FROM episodic_events "
            "WHERE user_id = ? AND status = 'ACTIVE' ORDER BY timestamp DESC LIMIT 500", (user_id,)).fetchall()
        conn.close()

        now = time.time()
        items = []
        for mid, content, importance, ts, metadata in rows:
            try:
                tags = set(json.loads(metadata or "{}").get("schema_tags") or [])
            except ValueError:
                tags = set()
            if not tags & self.PLAN_SCHEMAS:
                continue
            age_days = max(0.0, (now - float(ts)) / 86400)
            priority = (float(importance) / 10) * math.exp(-age_days / 14) * (1 + 0.5 * bool(tags & predicted))
            items.append((priority, mid, content))

        items.sort(reverse=True)
        return [{"step": i + 1, "memory_id": mid, "action": content, "priority": round(p, 3)}
                for i, (p, mid, content) in enumerate(items[:max_steps])]
