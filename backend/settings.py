"""
Central configuration for the Recallic backend.

Precedence (highest first): environment variables > config.yaml > defaults
documented in config.yaml. Secrets are read from the environment only.

Environment variables:
  RECALLIC_CONFIG         path to an alternative config.yaml
  RECALLIC_STORAGE_ROOT   storage directory (overrides storage.root)
  RECALLIC_ENGINE_URL     vector engine base URL (overrides engine.url)
  RECALLIC_INTERNAL_KEY   shared secret for the vector engine (required)
  RECALLIC_EMBEDDING_PROVIDER  overrides models.embedding_provider
  CORS_ALLOWED_ORIGINS    comma-separated origins (overrides api.cors_allowed_origins)
"""
import os
import threading
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_lock = threading.Lock()


class ConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def config():
    path = Path(os.environ.get("RECALLIC_CONFIG") or PROJECT_ROOT / "config.yaml")
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")


def reload():
    """Drop cached configuration (tests and tools that change env vars)."""
    config.cache_clear()


def get(path, default=None):
    """Dotted lookup into config.yaml, e.g. get("recall.candidate_pool")."""
    node = config()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def storage_root():
    root = os.environ.get("RECALLIC_STORAGE_ROOT") or get("storage.root", "backend/storage")
    root = Path(root)
    return root if root.is_absolute() else PROJECT_ROOT / root


def storage_path(*parts, is_dir=False):
    """Absolute path under the storage root; parent directories are created."""
    p = storage_root().joinpath(*parts)
    with _lock:
        (p if is_dir else p.parent).mkdir(parents=True, exist_ok=True)
    return str(p)


# ---------------------------------------------------------------------------
# Vector engine
# ---------------------------------------------------------------------------
def engine_url():
    return (os.environ.get("RECALLIC_ENGINE_URL") or get("engine.url") or "").rstrip("/")


def engine_timeout():
    return float(get("engine.timeout_seconds", 10))


def internal_key():
    key = os.environ.get("RECALLIC_INTERNAL_KEY", "")
    if len(key) < 16:
        raise ConfigError("RECALLIC_INTERNAL_KEY is not set (>= 16 chars). "
                          "Run `python -m backend.tools.setup_env` to create one in .env.")
    return key


def hnsw_params():
    h = get("hnsw", {}) or {}
    return {"M": int(h.get("M", 16)),
            "efConstruction": int(h.get("ef_construction", 100)),
            "efSearch": int(h.get("ef_search", 50))}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def cors_origins():
    env = os.environ.get("CORS_ALLOWED_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    return list(get("api.cors_allowed_origins", []) or [])
