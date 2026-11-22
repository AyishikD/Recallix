import sqlite3
import time

from backend import settings
from ..memory_ranking.reinforcement_learning import MemoryRL


class ForgettingWorker:
    """
    Forgetting policy (thresholds from config.yaml workers.forgetting):

      protect  importance >= protect_min_importance                  -> keep
      keep     well-reinforced (value = importance/10 * reinforcement/10 > 0.6) -> keep
      delete   never recalled, importance <= delete_max_importance, older than delete_min_age_days
      archive  importance <= archive_max_importance, older than archive_min_age_days
               -> copied to the archive store, then removed from active memory

    Removal goes through MemoryManager.delete, so the vector, long-term copy
    and timeline entry are removed too.
    """

    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.rl = MemoryRL()

    @staticmethod
    def _cfg():
        c = settings.get("workers.forgetting", {}) or {}
        return {"enabled": bool(c.get("enabled", True)),
                "delete_max_importance": float(c.get("delete_max_importance", 3.0)),
                "delete_min_age_days": float(c.get("delete_min_age_days", 7)),
                "archive_max_importance": float(c.get("archive_max_importance", 5.0)),
                "archive_min_age_days": float(c.get("archive_min_age_days", 30)),
                "protect_min_importance": float(c.get("protect_min_importance", 8.0))}

    def decide(self, memory, now, cfg):
        importance = float(memory["importance"] or 0.0)
        reinforcement = float(memory["reinforcement_score"] or 1.0)
        age_days = (now - float(memory["timestamp"])) / 86400
        if importance >= cfg["protect_min_importance"]:
            return "protect"
        if self.rl.calculate_value(importance, reinforcement) > 0.6:
            return "keep"
        if (importance <= cfg["delete_max_importance"] and (memory["retrieval_count"] or 0) == 0
                and age_days > cfg["delete_min_age_days"]):
            return "delete"
        if importance <= cfg["archive_max_importance"] and age_days > cfg["archive_min_age_days"]:
            return "archive"
        return "keep"

    def run_cleanup(self, user_id=None, now=None):
        cfg = self._cfg()
        if not cfg["enabled"]:
            return {"enabled": False}
        now = time.time() if now is None else now
        min_age = min(cfg["delete_min_age_days"], cfg["archive_min_age_days"]) * 86400
        conn = sqlite3.connect(self.memory.episodic.db_path, timeout=30)
        q = ("SELECT id, user_id, content, timestamp, importance, reinforcement_score, retrieval_count, metadata "
             "FROM episodic_events WHERE status = 'ACTIVE' AND timestamp < ?")
        args = [now - min_age]
        if user_id:
            q += " AND user_id = ?"
            args.append(user_id)
        cols = ("id", "user_id", "content", "timestamp", "importance", "reinforcement_score", "retrieval_count", "metadata")
        rows = [dict(zip(cols, r)) for r in conn.execute(q, args).fetchall()]
        conn.close()

        counts = {"evaluated": len(rows), "deleted": 0, "archived": 0}
        for m in rows:
            action = self.decide(m, now, cfg)
            if action == "delete":
                counts["deleted"] += self.memory.delete(m["user_id"], m["id"], reason="forgetting policy") is not None
            elif action == "archive":
                self.memory.archive.archive(m)
                counts["archived"] += self.memory.delete(m["user_id"], m["id"], reason="forgetting policy (archived first)") is not None
        return counts
