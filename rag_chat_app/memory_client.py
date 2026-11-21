import os

import requests


class MemoryOSClient:
    """
    Recallix client for the RAG chat demo. Uses one *service* key for the app and
    scopes every call to the logged-in chat user, so each user has a separate
    memory space.

      RECALLIX_SERVICE_KEY  python -m backend.tools.keys create --user rag-app --role service
      RECALLIX_API_URL      default http://localhost:8000
    """

    def __init__(self, user_id, api_key=None, base_url=None):
        self.user_id = user_id
        self.base_url = (base_url or os.environ.get("RECALLIX_API_URL", "http://localhost:8000")).rstrip("/")
        key = api_key or os.environ.get("RECALLIX_SERVICE_KEY")
        if not key:
            raise RuntimeError("Set RECALLIX_SERVICE_KEY to a Recallix service key.")
        self.headers = {"X-API-Key": key}

    def _request(self, method, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "POST":
                res = requests.post(url, json={**(data or {}), "user_id": self.user_id}, headers=self.headers, timeout=30)
            else:
                res = requests.get(url, params={"user_id": self.user_id}, headers=self.headers, timeout=30)
            return res.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def store(self, content):
        return self._request("POST", "/memory/store", {"content": content})

    def recall(self, query):
        return self._request("POST", "/memory/recall", {"query": query})

    def get_timeline(self):
        return self._request("GET", "/memory/timeline")

    def get_insights(self):
        return self._request("GET", "/memory/insights")

    def get_world_state(self):
        return self._request("GET", "/memory/world-state")

    def get_stats(self):
        return self._request("GET", "/memory/stats")

    def agent_store(self, content, agent_id, memory_type="private"):
        return self._request("POST", "/agents/store", {"content": content, "agent_id": agent_id, "memory_type": memory_type})

    def agent_recall(self, query, agent_id):
        return self._request("POST", "/agents/recall", {"query": query, "agent_id": agent_id})

    def list_agents(self):
        return self._request("GET", "/agents/list")

    def get_profile(self):
        return self._request("GET", "/memory/profile")
