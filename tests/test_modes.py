import pytest
from pydantic import ValidationError

from backend import settings
from backend.api.public import public_api
from backend.brain.features import aggregation_check, answer_verify, modes


def with_config(monkeypatch, **recall):
    real = settings.get

    def get(key, default=None):
        if key.startswith("recall.") and key[len("recall."):] in recall:
            return recall[key[len("recall."):]]
        return real(key, default)
    monkeypatch.setattr(settings, "get", get)


def test_mode_resolves_from_request_then_config_then_default(monkeypatch):
    assert modes.resolve() == "balanced"
    with_config(monkeypatch, mode="helpful")
    assert modes.resolve() == "helpful"
    assert modes.resolve("cautious") == "cautious"
    assert modes.resolve("reckless") == "balanced"


def test_each_mode_turns_on_only_its_own_feature(monkeypatch):
    with_config(monkeypatch, answer_check={}, aggregation_check={})
    for mode, verify, agg in (("balanced", False, False), ("cautious", True, False), ("helpful", False, True)):
        token = modes.use(mode)
        try:
            assert answer_verify.config()["enabled"] is verify
            assert aggregation_check.config()["enabled"] is agg
            assert aggregation_check.config()["mode"] == "rescue"
        finally:
            modes.reset(token)
    assert modes.current() == "balanced"


def test_a_mode_never_turns_a_configured_feature_off(monkeypatch):
    with_config(monkeypatch, answer_check={"verify": {"enabled": True}}, aggregation_check={"enabled": True, "mode": "decide"})
    token = modes.use("balanced")
    try:
        assert answer_verify.config()["enabled"] is True
        assert aggregation_check.config() == {**aggregation_check.config(), "enabled": True, "mode": "decide"}
    finally:
        modes.reset(token)
    token = modes.use("helpful")
    try:
        assert aggregation_check.config()["mode"] == "rescue"   # helpful means rescue, whatever the config says
    finally:
        modes.reset(token)


def test_the_api_accepts_only_known_modes():
    assert public_api.RecallRequest(query="q", mode="cautious").mode == "cautious"
    with pytest.raises(ValidationError):
        public_api.RecallRequest(query="q", mode="reckless")


def test_recall_reports_its_mode_and_restores_it(client, api_keys):
    h = api_keys["alice"]
    assert client.post("/memory/store", json={"content": "I moved to Pune in March."}, headers=h).status_code == 200
    out = client.post("/memory/recall", json={"query": "Where did I move?"}, headers=h).json()["data"]
    assert out["receipt"]["mode"] == "balanced" and out["debug"]["mode"] == "balanced"
    out = client.post("/memory/recall", json={"query": "Where did I move?", "mode": "helpful"}, headers=h).json()["data"]
    assert out["receipt"]["mode"] == "helpful"
    assert client.post("/memory/recall", json={"query": "x", "mode": "reckless"}, headers=h).status_code == 422
    assert modes.current() == "balanced"
