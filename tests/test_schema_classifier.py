import threading

import numpy as np

from backend.brain.features.schemas import SchemaClassifier
from tests.conftest import hash_embedding


def _embed(text):
    v = np.asarray(hash_embedding(text), dtype=np.float32)
    return v / np.sqrt(v @ v) if v.any() else None  # no np.linalg.norm: the test patches it


def test_request_during_first_build_never_sees_partial_state(tmp_path, monkeypatch):
    """
    Regression for a CI failure (`matmul ... has 0 dimensions`): a request that
    arrived while another was finishing the prototype centroids saw the seeds
    already set and multiplied with centroids that were still None.
    """
    clf = SchemaClassifier(_embed, "hash-test", cache_path=str(tmp_path / "protos.json"))
    text = "I need to renew my passport."
    vec = _embed(text)

    in_window, release = threading.Event(), threading.Event()
    real_norm = np.linalg.norm

    def pausing_norm(*a, **k):  # called while the centroids are being finished
        in_window.set()
        release.wait(timeout=5)
        return real_norm(*a, **k)
    monkeypatch.setattr(np.linalg, "norm", pausing_norm)

    first = {}
    t = threading.Thread(target=lambda: first.setdefault("label", clf.classify(text, vec)))
    t.start()
    assert in_window.wait(timeout=5)

    second = {}

    def concurrent():
        try:
            second["label"] = clf.classify(text, vec)
        except Exception as e:  # the bug
            second["error"] = repr(e)
    t2 = threading.Thread(target=concurrent)
    t2.start()
    t2.join(timeout=0.5)  # a correct implementation blocks here until the build finishes
    release.set()
    t.join(timeout=5)
    t2.join(timeout=5)

    assert "error" not in second, second.get("error")
    assert second["label"] == first["label"]
