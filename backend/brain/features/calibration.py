"""
A guarantee on wrong refusals for the answer check (no LLM, no new model).

Why: a decline threshold chosen on one dataset need not transfer to another. On
LongMemEval the LoCoMo-fitted threshold wrongly declined 20% of answerable questions
against a 5% target, while the answer-check *signal* still separated answerable
from unanswerable questions well (AUC 0.80). The fix is to calibrate the threshold
on the deployment's own questions, and to say how sure we are.

Calibration data: the answer-check margins `s_1..s_n` of *answerable* questions
whose served memories did contain the answer, taken from user feedback (see
backend/tools/calibrate.py) or a labelled sample.

The guarantee (one-sided binomial test, fixed-sequence "learn then test"):
for a threshold t, the number R of calibration margins below t is Binomial(n, p_t),
where p_t is the true wrong-refusal rate. If p_t > alpha then

    P(R <= r) <= BinomCDF(r; n, alpha)

so declaring "p_t <= alpha" whenever BinomCDF(R; n, alpha) <= delta is wrong with
probability at most delta. Candidate thresholds are tested from least to most
declining (p_t grows with t) and the last one that passes is used; stopping at the
first failure keeps the overall error probability at delta (Angelopoulos et al.,
"Learn then Test", 2021). Result: with probability >= 1 - delta over the
calibration sample, the wrong-refusal rate is <= alpha, *provided future questions
resemble the calibration questions* (exchangeability). That proviso is why the
calibration belongs to each deployment.

With too few calibration questions no threshold can be certified: (1 - alpha)^n
must be <= delta, i.e. n >= 59 for alpha = delta = 0.05. Then the result is
"never decline on this signal" (rate 0, trivially within alpha), stated as such.
"""
import json
import math

from backend import settings

CALIBRATION_FILE = "answer_check_calibration.json"


def _log_binom_pmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(r, n, p):
    """P(Binomial(n, p) <= r), summed in log space (stable for n in the tens of thousands)."""
    if r < 0:
        return 0.0
    if r >= n:
        return 1.0
    logs = [_log_binom_pmf(k, n, p) for k in range(r + 1)]
    top = max(logs)
    if top == -math.inf:
        return 0.0
    return min(1.0, math.exp(top) * sum(math.exp(x - top) for x in logs))


def min_calibration_size(alpha, delta):
    """Smallest n for which any declining threshold can be certified."""
    return math.ceil(math.log(delta) / math.log1p(-alpha))


def certify_threshold(margins, alpha=0.05, delta=0.05):
    """
    The most declining threshold whose wrong-refusal rate is certified <= alpha with
    confidence 1 - delta, from margins of answerable questions. Returns a certificate dict.
    `threshold` is None when nothing can be certified ("never decline on this signal").
    """
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be in (0, 1)")
    s = sorted(float(m) for m in margins if m is not None and math.isfinite(float(m)))
    n = len(s)
    # Candidate thresholds are the distinct margins, least declining first. A threshold t
    # declines a question when its margin < t, so t = u_j refuses every margin below u_j.
    threshold, refused = None, 0
    i = 0
    for u in sorted(set(s)):
        while i < n and s[i] < u:
            i += 1
        if binom_cdf(i, n, alpha) > delta:  # cannot certify this one: stop (fixed sequence)
            break
        threshold, refused = u, i
    certified = threshold is not None
    return {
        "method": "binomial learn-then-test",
        "alpha": alpha,
        "delta": delta,
        "n": n,
        "threshold": threshold,
        "calibration_refusal_rate": (refused / n) if n else None,
        "certified": certified,
        "min_n": min_calibration_size(alpha, delta),
        "statement": (f"wrong refusals <= {alpha:.0%} with probability >= {1 - delta:.0%}, "
                      "for questions like the calibration questions")
        if certified else
        f"not enough calibration data ({n} < {min_calibration_size(alpha, delta)}): the answer check will not decline",
    }


def save(certificate, model_name):
    path = settings.storage_path(CALIBRATION_FILE)
    with open(path, "w") as f:
        json.dump({**certificate, "model": model_name}, f, indent=2)
    return path


def load(model_name):
    """This deployment's certificate for `model_name`, or None."""
    try:
        with open(settings.storage_path(CALIBRATION_FILE)) as f:
            cert = json.load(f)
    except (OSError, ValueError):
        return None
    return cert if cert.get("model") == model_name else None
