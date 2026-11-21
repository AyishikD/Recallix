"""
HTTP client for the C++ vector engine.

* Base URL, timeout and the shared secret come from backend.settings
  (config.yaml / environment); nothing is hardcoded here.
* Every engine response carries X-Engine-Instance, which changes whenever the
  engine process restarts (its indexes are in memory). Registered listeners
  are called once per change so the caller can rebuild the indexes.
"""
import threading

import requests

from backend import settings


class EngineError(RuntimeError):
    pass


_session = requests.Session()
_listeners = []
_state = {"instance": None}
_state_lock = threading.Lock()


def on_instance_change(callback):
    """callback(old_instance, new_instance) — called from the requesting thread."""
    _listeners.append(callback)


def _track(response):
    instance = response.headers.get("X-Engine-Instance")
    if not instance:
        return
    with _state_lock:
        old = _state["instance"]
        if old == instance:
            return
        _state["instance"] = instance
    for cb in list(_listeners):
        cb(old, instance)


def _request(method, path, json_data=None):
    url = f"{settings.engine_url()}{path}"
    headers = {"X-Internal-Key": settings.internal_key()}
    try:
        res = _session.request(method, url, json=json_data, headers=headers, timeout=settings.engine_timeout())
    except requests.RequestException as e:
        raise EngineError(f"vector engine unreachable at {settings.engine_url()}: {e}") from e
    _track(res)
    if res.status_code != 200:
        try:
            detail = res.json().get("error", res.text)
        except ValueError:
            detail = res.text
        raise EngineError(f"{method} {path} -> {res.status_code}: {detail}")
    return res.json()


def engine_post(path, json_data=None):
    return _request("POST", path, json_data or {})


def engine_get(path):
    return _request("GET", path)


def current_instance():
    with _state_lock:
        return _state["instance"]
