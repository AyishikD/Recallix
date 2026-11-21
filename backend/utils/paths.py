import os
from pathlib import Path

from backend import settings

_STORAGE_PREFIX = "backend/storage/"


def get_project_root():
    """Absolute path to the project root."""
    return settings.PROJECT_ROOT


def get_db_path(relative_path="backend/storage/sqlite_db/memory.db"):
    """
    Absolute path for a storage file. Paths under "backend/storage/" resolve
    against the configured storage root (config.yaml storage.root or
    RECALLIX_STORAGE_ROOT) and their parent directory is created.
    """
    rel = relative_path.replace("\\", "/")
    if rel.startswith(_STORAGE_PREFIX):
        rest = rel[len(_STORAGE_PREFIX):]
        return settings.storage_path(*[p for p in rest.split("/") if p], is_dir=rel.endswith("/"))
    return str(Path(get_project_root()) / rel)


def ensure_dir(path):
    """Ensures the parent directory of a file path exists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
