from backend.brain.features import answer_check


class FakeChecker:
    """Says a memory answers the question iff it contains the word given at construction."""
    model_name = "fake"

    def __init__(self, needle, threshold=0.0):
        self.needle, self._t = needle, threshold

    def threshold(self):
        return self._t

    def check(self, question, contexts):
        return [{"margin": 1.0 if self.needle in c else -1.0, "answer": self.needle if self.needle in c else ""}
                for c in contexts]


def _store(client, h, text):
    assert client.post("/memory/store", json={"content": text}, headers=h).status_code == 200


def test_abstains_when_no_served_memory_answers(client, api_keys, monkeypatch):
    h = api_keys["alice"]
    _store(client, h, "Project Heron deadline is April 3.")
    _store(client, h, "Project Egret deadline is July 19.")

    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: FakeChecker("April")))
    out = client.post("/memory/recall", json={"query": "When is Project Heron due?"}, headers=h).json()["data"]
    assert out["status"] == "CONSISTENT"
    top = out["memories"][0]
    assert top["answer"] == "April" and top["answer_margin"] == 1.0
    assert out["debug"]["answer_check"]["best_margin"] == 1.0

    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: FakeChecker("October")))
    out = client.post("/memory/recall", json={"query": "When is Project Heron due?"}, headers=h).json()["data"]
    assert out["status"] == "ABSTAINED" and out["memories"] == []


def test_a_decline_keeps_the_results_only_when_configured(client, api_keys, monkeypatch):
    from backend import settings
    h = api_keys["alice"]
    _store(client, h, "Project Heron deadline is April 3.")
    _store(client, h, "Project Egret deadline is July 19.")
    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: FakeChecker("October")))
    real = settings.get
    monkeypatch.setattr(settings, "get", lambda path, default=None:
                        True if path == "recall.keep_results_on_decline" else real(path, default))
    out = client.post("/memory/recall", json={"query": "When is Project Heron due?"}, headers=h).json()["data"]
    assert out["status"] == "ABSTAINED" and out["answer"] is None   # the decision is unchanged
    assert "Heron" in out["memories"][0]["content"]                 # but the ranked results come back


def test_keep_on_decline_is_a_per_request_choice(client, api_keys, monkeypatch):
    h = api_keys["alice"]
    _store(client, h, "Project Heron deadline is April 3.")
    monkeypatch.setattr(answer_check.AnswerChecker, "from_config", classmethod(lambda cls: FakeChecker("October")))
    ask = {"query": "When is Project Heron due?"}
    out = client.post("/memory/recall", json={**ask, "keep_on_decline": True}, headers=h).json()["data"]
    assert out["status"] == "ABSTAINED" and out["answer"] is None and "Heron" in out["memories"][0]["content"]
    out = client.post("/memory/recall", json=ask, headers=h).json()["data"]                      # not asked: nothing
    assert out["status"] == "ABSTAINED" and out["memories"] == []
    out = client.post("/agents/recall", json={**ask, "agent_id": "default_agent", "keep_on_decline": True}, headers=h).json()["data"]
    assert out["status"] == "ABSTAINED" and out["memories"]


def test_threshold_resolution(monkeypatch, tmp_path):
    """Override > value fitted for the same model > the model's own boundary (0)."""
    from backend import settings
    chk = answer_check.AnswerChecker.__new__(answer_check.AnswerChecker)
    chk.model_name = "m1"
    fitted = tmp_path / "fitted.json"
    fitted.write_text('{"model": "m1", "threshold": 2.5}')
    monkeypatch.setattr(answer_check, "DEFAULT_PATH", str(fitted))
    real_get = settings.get
    monkeypatch.setattr(settings, "get", lambda p, d=None: {"enabled": True, "model": "m1"} if p == "recall.answer_check" else real_get(p, d))
    assert chk.threshold() == 2.5
    chk.model_name = "other-model"
    assert chk.threshold() == 0.0
    monkeypatch.setattr(settings, "get", lambda p, d=None: {"enabled": True, "threshold": -1} if p == "recall.answer_check" else real_get(p, d))
    assert chk.threshold() == -1.0
