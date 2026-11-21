import threading
import time
from collections import deque

from backend import settings


class SensoryMemory:
    """Per-user in-process ring buffer of the most recent raw messages."""

    def __init__(self, ttl=None, size=None):
        self.ttl = float(ttl if ttl is not None else settings.get("memory.sensory_ttl_seconds", 60))
        self.size = int(size if size is not None else settings.get("memory.sensory_size", 10))
        self.memories = {}
        self._lock = threading.Lock()

    def add(self, user_id, message):
        with self._lock:
            self.memories.setdefault(user_id, deque(maxlen=self.size)).append(
                {"content": message, "timestamp": time.time()})

    def get(self, user_id):
        now = time.time()
        with self._lock:
            return [m for m in self.memories.get(user_id, ()) if now - m["timestamp"] < self.ttl]

    def clear_user(self, user_id):
        with self._lock:
            self.memories.pop(user_id, None)

    def clear_all(self):
        with self._lock:
            self.memories.clear()
