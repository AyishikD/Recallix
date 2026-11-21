import math
import time

from backend import settings


class DecayLogic:
    @staticmethod
    def calculate_strength(base_importance, last_accessed, access_count, now=None):
        """
        Ebbinghaus forgetting curve: strength = importance * exp(-t / S).

        Stability S grows with importance and (logarithmically) with the number
        of accesses, and is measured in units of memory.decay_unit_hours
        (config.yaml), applied consistently at every age.
        """
        now = time.time() if now is None else now
        if last_accessed is None:
            last_accessed = now
        t_hours = max(0.0, now - float(last_accessed)) / 3600.0
        unit = float(settings.get("memory.decay_unit_hours", 24))
        stability = max((float(base_importance) * 0.5) * (1 + math.log((access_count or 0) + 1)), 0.1)
        return float(base_importance) * math.exp(-t_hours / (stability * unit))

    @staticmethod
    def reinforce(access_count):
        return access_count + 1
