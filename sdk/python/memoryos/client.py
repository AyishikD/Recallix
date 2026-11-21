from typing import Any, Dict, Optional

import requests


class RecallixError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class MemoryClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", timeout: float = 30):
        if not api_key:
            raise ValueError("api_key is required")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "Content-Type": "application/json"})

    def request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                params: Optional[Dict] = None) -> Dict[str, Any]:
        """Returns the response's `data`. Raises RecallixError on HTTP errors."""
        body = {k: v for k, v in data.items() if v is not None} if data else None
        res = self.session.request(method, f"{self.base_url}{endpoint}", json=body, params=params, timeout=self.timeout)
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise RecallixError(res.status_code, str(detail))
        return res.json().get("data")
