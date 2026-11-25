"""
Authentication and authorisation dependencies.

  verify_api_key   X-API-Key -> principal {user_id, role, ...}; 401 if missing, 403 if unknown
  acting_user      the user a request operates on: the key's own user, or —
                   for service/admin keys only — an explicit user_id
  require_admin    admin role, for /internal/*
  rate_limit_writes  per-user write rate limit (config api.writes_per_second_per_user)
  rate_limit_reads   per-user recall rate limit (config api.reads_per_second_per_user); deep recall costs more
  check_id         user and agent ids: 1-128 characters, no control characters
  one_import_at_a_time  a user's imports run one after another (each can hold thousands of memories)
"""
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

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


ID_PATTERN = r"^[^\x00-\x1f\x7f]{1,128}$"  # also used by the request models
_ID_RE = re.compile(ID_PATTERN)


def check_id(value, what="user_id"):
    """User and agent ids end up in index names, SQL rows and logs: bounded, printable."""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise HTTPException(status_code=422, detail=f"{what} must be 1-128 characters with no control characters")
    return value


def acting_user(principal, requested_user_id=None):
    if requested_user_id:
        check_id(requested_user_id)
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
    """Sliding one-second window per user. Users idle for a second are dropped once many are tracked."""
    PRUNE_ABOVE = 10000

    def __init__(self):
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, user_id, limit, cost=1):
        now = time.monotonic()
        with self._lock:
            if len(self._events) > self.PRUNE_ABOVE:
                for uid in [u for u, q in self._events.items() if not q or now - q[-1] >= 1.0]:
                    del self._events[uid]
            q = self._events[user_id]
            while q and now - q[0] >= 1.0:
                q.popleft()
            if len(q) + cost > limit:
                return False
            q.extend([now] * cost)
            return True


_limiter = _RateLimiter()
_read_limiter = _RateLimiter()


def rate_limit_writes(user_id):
    limit = int(settings.get("api.writes_per_second_per_user", 50))
    if limit > 0 and not _limiter.allow(user_id, limit):
        raise HTTPException(status_code=429, detail=f"More than {limit} writes per second for this user")


def rate_limit_reads(user_id, deep=False):
    """Recall runs the answer check (and, for deep recall, a re-ranker): a deep recall counts as 4."""
    limit = int(settings.get("api.reads_per_second_per_user", 20))
    if limit > 0 and not _read_limiter.allow(user_id, limit, cost=4 if deep else 1):
        raise HTTPException(status_code=429, detail=f"More than {limit} recalls per second for this user "
                                                    "(a deep recall counts as 4)")


_importing = set()
_import_lock = threading.Lock()


@contextmanager
def one_import_at_a_time(user_id):
    with _import_lock:
        if user_id in _importing:
            raise HTTPException(status_code=429, detail="An import for this user is already running")
        _importing.add(user_id)
    try:
        yield
    finally:
        with _import_lock:
            _importing.discard(user_id)
