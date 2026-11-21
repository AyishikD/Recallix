"""
Authentication and authorisation dependencies.

  verify_api_key   X-API-Key -> principal {user_id, role, ...}; 401 if missing, 403 if unknown
  acting_user      the user a request operates on: the key's own user, or —
                   for service/admin keys only — an explicit user_id
  require_admin    admin role, for /internal/*
  rate_limit_writes  per-user write rate limit (config api.writes_per_second_per_user)
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request

from backend import settings
from backend.api import keys


async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(status_code=401, detail="API key missing (X-API-Key header)")
    principal = keys.lookup(key)
    if principal is None:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return principal


def acting_user(principal, requested_user_id=None):
    if requested_user_id and requested_user_id != principal["user_id"]:
        if principal.get("role") not in ("service", "admin"):
            raise HTTPException(status_code=403, detail="This key may only access its own user")
        return requested_user_id
    return principal["user_id"]


async def require_admin(principal=Depends(verify_api_key)):
    if principal.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin key required")
    return principal


class _RateLimiter:
    """Sliding one-second window per user."""

    def __init__(self):
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, user_id, limit):
        now = time.monotonic()
        with self._lock:
            q = self._events[user_id]
            while q and now - q[0] >= 1.0:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


_limiter = _RateLimiter()


def rate_limit_writes(user_id):
    limit = int(settings.get("api.writes_per_second_per_user", 50))
    if limit > 0 and not _limiter.allow(user_id, limit):
        raise HTTPException(status_code=429, detail=f"More than {limit} writes per second for this user")
