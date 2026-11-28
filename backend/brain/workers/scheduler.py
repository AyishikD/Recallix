"""
Background jobs, run in one daemon thread inside the API process.

  state_inference  world-state from each active user's recent memories
  prediction       Markov next-topic prediction per active user
  replay           reflection from each active user's long-term facts
  forgetting       forgetting policy over all memories
  learning         refit the ranker from feedback; kept only if better on held-out feedback
                   (backend/brain/memory_ranking/self_improve.py)

Intervals and the "active user" window come from config.yaml `workers`.
Every job can also be run on demand (POST /internal/jobs/run, tests).
"""
import threading
import time
import traceback

from backend import settings
from .forgetting_worker import ForgettingWorker
from .replay_worker import ReplayWorker

_INTERVAL_KEYS = {
    "state_inference": ("workers.state_inference_minutes", 15, 60),
    "prediction": ("workers.prediction_minutes", 30, 60),
    "replay": ("workers.replay_hours", 6, 3600),
    "forgetting": ("workers.forgetting_hours", 24, 3600),
    "learning": ("workers.learning_hours", 24, 3600),
}


class Scheduler:
    def __init__(self, memory_manager):
        self.mm = memory_manager
        self.replay = ReplayWorker(memory_manager)
        self.forgetting = ForgettingWorker(memory_manager)
        self._stop = threading.Event()
        self._thread = None
        self._last_run = {}
        self.last_results = {}

    # ------------------------------------------------------------------
    def interval_seconds(self, job):
        key, default, unit = _INTERVAL_KEYS[job]
        return float(settings.get(key, default)) * unit

    def active_users(self):
        days = float(settings.get("workers.active_user_days", 30))
        return self.mm.episodic.active_users(time.time() - days * 86400)

    def run_job(self, job, user_id=None):
        users = [user_id] if user_id else None
        if job == "forgetting":
            result = self.forgetting.run_cleanup(user_id=user_id)
        elif job == "learning":  # deployment-wide: one ranker and one answer-check threshold for all users
            from backend.brain.features import answer_check_learning
            from backend.brain.memory_ranking import self_improve
            result = self_improve.run()
            result["answer_check"] = answer_check_learning.run()
        else:
            users = users or self.active_users()
            result = {}
            for uid in users:
                if job == "state_inference":
                    result[uid] = self.mm.state_inference.infer_and_update(uid, self.mm.episodic.list_recent(uid, 200))
                elif job == "prediction":
                    result[uid] = self.mm.prediction_engine.predict(uid)
                elif job == "replay":
                    result[uid] = self.replay.consolidate(uid)
                else:
                    raise ValueError(f"unknown job {job!r}")
        self._last_run[job] = time.time()
        self.last_results[job] = result
        return result

    # ------------------------------------------------------------------
    def start(self):
        if not settings.get("workers.enabled", True) or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="recallic-scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        now = time.time()
        for job in _INTERVAL_KEYS:  # first run one interval after startup
            self._last_run.setdefault(job, now)
        while not self._stop.wait(timeout=30):
            for job in _INTERVAL_KEYS:
                if time.time() - self._last_run[job] >= self.interval_seconds(job):
                    try:
                        self.run_job(job)
                    except Exception:
                        print(f"[Scheduler] job {job} failed:\n{traceback.format_exc()}")
                        self._last_run[job] = time.time()
