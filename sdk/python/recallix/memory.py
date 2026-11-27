from typing import Any, Dict, List, Optional

from .client import MemoryClient, RecallixError


class Memory:
    """
    Recallix client.

        mem = Memory(api_key=os.environ["RECALLIX_API_KEY"])
        mem.store("I live in Pune.")
        hits = mem.recall("Where do I live?")

    `user_id` may only be passed with a service/admin key (multi-user apps);
    a user key always acts as its own user.
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", timeout: float = 30):
        self.client = MemoryClient(api_key, base_url, timeout)

    def store(self, content: str, agent_id: str = "default_agent", memory_type: str = "private",
              user_id: Optional[str] = None, metadata: Optional[Dict] = None,
              timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        memory_type: "private" (only this agent) or "shared" (all of the user's agents).
        timestamp: when the message was originally said (Unix seconds; server 0.2.5+). Default: now.
        """
        return self.client.request("POST", "/memory/store", data={
            "content": content, "agent_id": agent_id, "memory_type": memory_type,
            "user_id": user_id, "metadata": metadata, "timestamp": timestamp})

    BATCH_SIZE = 100  # server api.max_batch_items

    def store_many(self, items: List[Any], agent_id: str = "default_agent", memory_type: str = "private",
                   user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stores several messages in as few requests as possible (server 0.2.6+; an older server gets one store
        per message). Each item is a string or {"content", "timestamp"?, "metadata"?, "memory_type"?}.
        Same result as storing them one by one in time order; repeats are skipped.
        Returns {"stored", "skipped_duplicates", "results"}, with `results` in the order of `items`.
        """
        rows = [{"content": it} if isinstance(it, str) else dict(it) for it in items]
        results: List[Dict[str, Any]] = []
        for start in range(0, len(rows), self.BATCH_SIZE):
            chunk = rows[start:start + self.BATCH_SIZE]
            try:
                out = self.client.request("POST", "/memory/store_batch", data={
                    "items": chunk, "agent_id": agent_id, "memory_type": memory_type, "user_id": user_id})
                results.extend(out["results"])
            except RecallixError as e:
                if e.status not in (404, 405):
                    raise
                for row in sorted(chunk, key=lambda r: (r.get("timestamp") is None, r.get("timestamp") or 0)):
                    row["_result"] = self.store(row["content"], agent_id=agent_id,
                                                memory_type=row.get("memory_type") or memory_type, user_id=user_id,
                                                metadata=row.get("metadata"), timestamp=row.get("timestamp"))
                results.extend(row.pop("_result") for row in chunk)
        stored = sum(r.get("status") == "stored" for r in results)
        return {"stored": stored, "skipped_duplicates": len(results) - stored, "results": results}

    def recall(self, query: str, limit: int = 10, agent_id: str = "default_agent",
               user_id: Optional[str] = None, deep: Optional[bool] = None,
               keep_on_decline: Optional[bool] = None, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        mode: how readily recall answers. "cautious" verifies every answer (more "I don't know", fewer wrong
        answers), "balanced" is the default, "helpful" answers more counting / ordering / time questions (fewer
        refusals). Default: server setting (recall.mode).

        deep=True re-reads the candidates with a re-ranker: better when the answer is spread over
        several memories ("what activities has she done with her kids?"), but slower. Default: server setting.

        keep_on_decline=True: when recall declines (status "ABSTAINED"), still return the results it
        found, as low-confidence ("answer" stays None). Check out["status"] before using them as an
        answer. Default: server setting (off).
        """
        data = {"query": query, "limit": limit, "agent_id": agent_id, "user_id": user_id}
        if deep is not None:
            data["deep"] = deep
        if keep_on_decline is not None:
            data["keep_on_decline"] = keep_on_decline
        if mode is not None:
            data["mode"] = mode
        return self.client.request("POST", "/memory/recall", data=data)

    def feedback(self, recall_id: str, relevant_memory_ids: List[int], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Tell Recallix which recalled memories were useful (trains the ranker)."""
        return self.client.request("POST", "/memory/feedback", data={
            "recall_id": recall_id, "relevant_memory_ids": relevant_memory_ids, "user_id": user_id})

    def used(self, recall: Dict[str, Any], *memories: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        One-line feedback after a recall, so Recallix learns from use:

            out = mem.recall("Where do I live?")
            mem.used(out)                       # the first memory answered it
            mem.used(out, out["memories"][2])   # or name the memories (or ids) your app actually used

        Recallix refits its ranking from this on a schedule and keeps a new fit only if it is better.
        """
        chosen = memories or recall.get("memories", [])[:1]
        ids = [int(m["id"] if isinstance(m, dict) else m) for m in chosen]
        return self.feedback(recall["recall_id"], ids, user_id=user_id)

    def none_used(self, recall: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Feedback that nothing in this recall was useful."""
        return self.feedback(recall["recall_id"], [], user_id=user_id)

    def proactive(self, message: str, agent_id: str = "default_agent", limit: int = 5,
                  user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Memories worth keeping in mind for a new message (no question needed), each with the reason."""
        return self.client.request("POST", "/memory/proactive", data={
            "message": message, "agent_id": agent_id, "limit": limit, "user_id": user_id})

    def export(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """All memories in the open "recallix-memory" format (portable to another Recallix)."""
        return self.client.request("GET", "/memory/export", params=_params(user_id=user_id))

    def import_memories(self, data: Any, format: str = "recallix", agent_id: str = "default_agent",
                        memory_type: str = "private", user_id: Optional[str] = None) -> Dict[str, Any]:
        """Import a Recallix export, a Mem0 export (format="mem0") or text, one memory per line (format="text")."""
        return self.client.request("POST", "/memory/import", data={
            "format": format, "data": data, "agent_id": agent_id, "memory_type": memory_type, "user_id": user_id})

    def delete(self, memory_id: int, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("DELETE", f"/memory/{memory_id}", params={"user_id": user_id} if user_id else None)

    def timeline(self, limit: int = 50, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/timeline", params=_params(limit=limit, user_id=user_id))

    def insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/insights", params=_params(user_id=user_id))

    def world_state(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/world-state", params=_params(user_id=user_id))

    def prediction(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/prediction", params=_params(user_id=user_id))

    def plan(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/plan", params=_params(user_id=user_id))

    def profile(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/profile", params=_params(user_id=user_id))

    def stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("GET", "/memory/stats", params=_params(user_id=user_id))


def _params(**kw):
    return {k: v for k, v in kw.items() if v is not None} or None
