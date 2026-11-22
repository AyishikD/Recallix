"""
Trust score: one calibrated probability that a served memory actually answers the question.

Recallix already computes several signals per served memory — meaning relevance, the
reader's answer margin, whether an answer span was found, the who-said-what contrast, and
how likely its facts are to still be true. A logistic model combines them into

    trust = P(this memory holds the answer | signals)

fitted on labelled questions (backend/tools/fit_trust.py: LoCoMo dev conversations, where
the evidence turns are known and adversarial questions have no right answer) and checked for
calibration out of sample. Agents can act on it: answer, hedge, or ask. If no fitted model is
installed, no trust is reported — the score is never invented.

Calibration does not transfer between kinds of conversation: on LoCoMo's held-out
conversations it is well calibrated, on LongMemEval it still ranks answers but its
probabilities are about half the true rate (eval/results/trust_check_*.json). So each
deployment can recalibrate it on its own confirmed answers (backend/tools/calibrate_trust.py):
the fitted model's log-odds z are rescaled as sigmoid(a*z + b) — two numbers — which keeps the
ranking and fixes the level. trust_calibration() says which calibration a score carries.
"""
import json
import math
import os

from backend import settings

FEATURES = ["relevance", "answer_margin", "has_answer_check", "answer_found", "subject_contrast", "has_contrast",
            "still_true"]
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "trust.default.json")


def features(memory):
    margin = memory.get("answer_margin")
    contrast = memory.get("subject_contrast")
    fresh = [f.get("freshness", {}).get("still_true") for f in (memory.get("receipt") or {}).get("facts", [])
             if f.get("freshness") and f["freshness"].get("still_true") is not None]
    return {
        "relevance": float(memory.get("relevance") or 0.0),
        "answer_margin": max(-10.0, min(10.0, float(margin))) / 10.0 if margin is not None else 0.0,
        "has_answer_check": 1.0 if margin is not None else 0.0,
        "answer_found": 1.0 if memory.get("answer") else 0.0,
        "subject_contrast": max(-10.0, min(10.0, float(contrast))) / 10.0 if contrast is not None else 0.0,
        "has_contrast": 1.0 if contrast is not None else 0.0,
        "still_true": min(fresh) if fresh else 1.0,
    }


CALIBRATION_FILE = "trust_calibration.json"
_model = {"path": None, "weights": None}
_deployment = {"mtime": None, "data": None}


def model():
    path = (settings.get("recall.trust", {}) or {}).get("model") or DEFAULT_PATH
    if _model["path"] != path:
        try:
            with open(path) as f:
                _model["weights"] = json.load(f).get("weights")
        except (OSError, ValueError):
            _model["weights"] = None
        _model["path"] = path
    return _model["weights"]


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


def logit(memory):
    """The fitted model's log-odds for this memory, or None when no model is installed."""
    w = model()
    if not w or not (settings.get("recall.trust", {}) or {}).get("enabled", True):
        return None
    x = features(memory)
    return w.get("bias", 0.0) + sum(w.get(k, 0.0) * v for k, v in x.items())


def deployment_calibration():
    """This deployment's recalibration {"a", "b", "n", ...} (calibrate_trust.py), or None."""
    path = settings.storage_path(CALIBRATION_FILE)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _deployment.update(mtime=None, data=None)
        return None
    if _deployment["mtime"] != mtime:
        try:
            with open(path) as f:
                data = json.load(f)
            _deployment["data"] = data if data.get("model_sha256") == model_sha256() else None
        except (OSError, ValueError):
            _deployment["data"] = None
        _deployment["mtime"] = mtime
    return _deployment["data"]


def model_sha256():
    import hashlib
    path = (settings.get("recall.trust", {}) or {}).get("model") or DEFAULT_PATH
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def save_calibration(a, b, report):
    path = settings.storage_path(CALIBRATION_FILE)
    with open(path, "w") as f:
        json.dump({"a": a, "b": b, "model_sha256": model_sha256(), **report}, f, indent=2)
    return path


def trust_calibration():
    """Which calibration trust scores carry: {"source", "n"?}."""
    if not model():
        return None
    cal = deployment_calibration()
    if cal:
        return {"source": "this deployment", "n": cal.get("n"), "fitted_at": cal.get("fitted_at")}
    return {"source": "reference (LoCoMo); not yet calibrated for this deployment — treat as a relative score"}


def score(memory):
    """Calibrated probability in [0, 1], or None when no fitted model is installed."""
    z = logit(memory)
    if z is None:
        return None
    cal = deployment_calibration()
    if cal:
        z = cal["a"] * z + cal["b"]
    return round(_sigmoid(z), 3)


def fit_recalibration(logits, labels, prior_strength=1.0, iters=50):
    """
    Platt rescaling of the fitted model's log-odds: (a, b) maximising the likelihood of
    sigmoid(a*z + b), with a Gaussian prior centred on the identity (a=1, b=0) so that a few
    labels move it only a little. Newton's method on two parameters.
    """
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga, gb = prior_strength * (a - 1.0), prior_strength * b
        haa, hab, hbb = prior_strength, 0.0, prior_strength
        for z, y in zip(logits, labels):
            p = _sigmoid(a * z + b)
            r, w = p - y, p * (1.0 - p)
            ga += r * z
            gb += r
            haa += w * z * z
            hab += w * z
            hbb += w
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-9:
            break
    return a, b


def recalibrated(z, a, b):
    return _sigmoid(a * z + b)
