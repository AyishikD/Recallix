"""
API key store.

Keys are random tokens ("rx_" + 43 url-safe chars) shown once at creation and
stored only as SHA-256 hashes in <storage>/api_keys.json:

  {"version": 2, "keys": {"<sha256>": {"user_id", "role", "label", "created_at", "prefix"}}}

Roles:
  user     acts only as its own user_id
  service  may act on behalf of any user_id (multi-user apps, e.g. the RAG chat)
  admin    service rights + /internal/* maintenance endpoints

A legacy file mapping plaintext keys to {"user_id"} is migrated in place
(keys are hashed; their role becomes "user").
"""
import hashlib
import json
import os
import re
import secrets
import threading
import time

from backend.utils.paths import get_db_path

ROLES = ("user", "service", "admin")
_lock = threading.Lock()
_cache = {"mtime": None, "keys": {}}


def keys_path():
    return get_db_path("backend/storage/api_keys.json")


def hash_key(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _write(keys):
    path = keys_path()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": 2, "keys": keys}, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _load():
    path = keys_path()
    if not os.path.exists(path):
        return {}
    mtime = (path, os.path.getmtime(path))
    if _cache["mtime"] == mtime:
        return _cache["keys"]
    with open(path) as f:
        data = json.load(f)
    keys = data.get("keys", {})
    if data.get("version") != 2:  # legacy plaintext keys -> hashes
        keys = {(k if re.fullmatch(r"[0-9a-f]{64}", k) else hash_key(k)):
                {"user_id": v.get("user_id"), "role": v.get("role", "user"), "label": "migrated",
                 "created_at": time.time(), "prefix": k[:6]}
                for k, v in keys.items()}
        _write(keys)
        mtime = (path, os.path.getmtime(path))
    _cache.update(mtime=mtime, keys=keys)
    return keys


def lookup(key):
    """Principal dict for a presented key, or None."""
    if not key:
        return None
    with _lock:
        entry = _load().get(hash_key(key))
    return dict(entry) if entry else None


def create(user_id, role="user", label=""):
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if not user_id:
        raise ValueError("user_id is required")
    if len(user_id) > 128 or re.search(r"[\x00-\x1f\x7f]", user_id):
        raise ValueError("user_id must be at most 128 characters with no control characters")
    key = "rx_" + secrets.token_urlsafe(32)
    with _lock:
        keys = dict(_load())
        keys[hash_key(key)] = {"user_id": user_id, "role": role, "label": label,
                               "created_at": time.time(), "prefix": key[:8]}
        _write(keys)
        _cache["mtime"] = None
    return key


def revoke(prefix_or_hash):
    with _lock:
        keys = dict(_load())
        victims = [h for h, v in keys.items() if h == prefix_or_hash or v.get("prefix") == prefix_or_hash]
        for h in victims:
            del keys[h]
        if victims:
            _write(keys)
            _cache["mtime"] = None
    return len(victims)


def list_keys():
    with _lock:
        return [{"prefix": v.get("prefix"), "user_id": v.get("user_id"), "role": v.get("role"),
                 "label": v.get("label"), "created_at": v.get("created_at")} for v in _load().values()]
