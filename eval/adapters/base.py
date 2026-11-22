"""
The contract every memory system implements to be benchmarked by eval/harness.py.

Small on purpose, so any system — ours, a competitor's, a vector database — can be
plugged in and scored identically. A system receives each conversation turn with a
stable turn id and must return, for a question, the ids of the turns it would put in
front of an LLM, best first. Returning [] means "nothing relevant / I decline".

Vendors are invited to submit or correct their own adapter: the harness, datasets and
scoring are open, so every configuration is visible and every number reproducible.
"""


class MemorySystem:
    name = "base"
    #: shown in results so readers know exactly what ran (model names, settings, versions)
    config: dict = {}

    def add_conversation(self, user, turns):
        """Ingest one conversation. turns: [(turn_id, text, speaker, timestamp_seconds)] in order."""
        raise NotImplementedError

    def search(self, user, query, k):
        """Turn ids that answer `query`, best first, at most k. [] = decline."""
        raise NotImplementedError

    def close(self):
        pass
