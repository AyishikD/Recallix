from typing import Any, Dict, List, Optional

from .client import MemoryClient


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
              user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """memory_type: "private" (only this agent) or "shared" (all of the user's agents)."""
        return self.client.request("POST", "/memory/store", data={
            "content": content, "agent_id": agent_id, "memory_type": memory_type,
            "user_id": user_id, "metadata": metadata})

    def recall(self, query: str, limit: int = 10, agent_id: str = "default_agent",
               user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.client.request("POST", "/memory/recall", data={
            "query": query, "limit": limit, "agent_id": agent_id, "user_id": user_id})

    def feedback(self, recall_id: str, relevant_memory_ids: List[int], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Tell Recallix which recalled memories were useful (trains the ranker)."""
        return self.client.request("POST", "/memory/feedback", data={
            "recall_id": recall_id, "relevant_memory_ids": relevant_memory_ids, "user_id": user_id})

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
