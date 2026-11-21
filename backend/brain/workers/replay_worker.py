import time
from collections import Counter

from backend.brain.features.summarize import select


class ReplayWorker:
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.last_replay = 0
        self.replay_interval = 6 * 60 * 60  # 6 hours

    def consolidate(self, user_id):
        """
        Memory replay ("sleep cycle"), no LLM.

        1. Take the user's retained long-term facts (strength-weighted).
        2. Pick the most representative, non-redundant ones (MMR over embeddings).
        3. Describe what they are about via the schema distribution.
        4. Store that as a reflection, with confidence = mean normalised strength.
        """
        lt_memories = self.memory.long_term.get(user_id)[:50]
        if not lt_memories:
            return None

        facts = [m["fact"] for m in lt_memories]
        embed = self.memory.semantic.embed
        key_idx = select(facts, embed, k=min(5, len(facts)))

        themes = Counter()
        for fact in facts:
            for label in self.memory.schema_classifier.classify(fact, embed(fact)):
                themes[label] += 1
        total = sum(themes.values()) or 1
        theme_str = ", ".join(f"{label} {count / total:.0%}" for label, count in themes.most_common(3))

        insight = f"Recurring themes: {theme_str}. Key facts: " + " | ".join(facts[i] for i in key_idx)
        strengths = [m.get("strength", 0.0) / 10.0 for m in lt_memories]
        confidence = round(min(1.0, sum(strengths) / len(strengths)), 3)

        self.memory.reflective.store_reflection(user_id, insight, [m.get("id", 0) for m in lt_memories[:10]], confidence)
        self.last_replay = time.time()
        return insight
