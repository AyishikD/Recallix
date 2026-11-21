import os
import json
import time

from backend.brain.features.summarize import extractive_summary
from backend.utils.paths import get_db_path


class CompressionWorker:
    def __init__(self, storage_path=None, embed_fn=None):
        """embed_fn: optional text -> unit vector (e.g. SemanticMemory.embed)."""
        self.storage_path = storage_path or get_db_path("backend/storage/conversation_summaries/")
        self.embed_fn = embed_fn
        os.makedirs(self.storage_path, exist_ok=True)

    def compress_conversations(self, user_id, messages):
        """
        Level 1: Conversation Compression (extractive, no LLM).
        Keeps the ~sqrt(n) most central, least redundant messages (MMR).
        """
        if not messages:
            return
        summary = extractive_summary(messages, self.embed_fn)

        filename = f"{user_id}_{int(time.time())}.json"
        with open(os.path.join(self.storage_path, filename), 'w') as f:
            json.dump({
                "user_id": user_id,
                "summary": summary,
                "timestamp": time.time(),
                "message_count": len(messages),
                "method": "extractive_mmr",
            }, f)
        return summary

    def compress_topics(self, user_id):
        """Level 2: summarise this user's conversation summaries the same way."""
        summaries = []
        for name in sorted(os.listdir(self.storage_path)):
            if name.startswith(f"{user_id}_") and name.endswith(".json"):
                with open(os.path.join(self.storage_path, name)) as f:
                    summaries.append(json.load(f).get("summary", ""))
        return extractive_summary(summaries, self.embed_fn) if summaries else ""
