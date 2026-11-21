"""
MAP logistic regression (numpy only), shared by the importance and ranking models.

    minimise  mean_i logloss(sigmoid(w · x_i), y_i)  +  (lam / n) * ||w - w0||^2

x_i = [1, f_1, ..., f_k] (bias first). w0 are prior weights: they keep fits
stable on small datasets and pull rarely-active features back to sensible
values instead of letting them drift. The objective is strictly convex, so
Newton's method (IRLS) reaches the unique optimum in a few iterations.
"""
import numpy as np


def design_matrix(feature_dicts, features):
    return np.array([[1.0] + [float(f.get(k, 0.0)) for k in features] for f in feature_dicts], dtype=float)


def fit(feature_dicts, labels, prior, features, lam=0.5, max_iter=50, tol=1e-8, sample_weight=None):
    """
    prior: dict with "bias" and every name in `features`. Returns a weight dict.
    sample_weight: optional per-example weights (normalised to mean 1), e.g. to
    let each data source count equally regardless of its size.
    """
    keys = ["bias"] + list(features)
    X = design_matrix(feature_dicts, features)
    y = np.asarray(labels, dtype=float)
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    sw = sw / sw.mean() if len(sw) else sw
    w0 = np.array([prior[k] for k in keys], dtype=float)
    w = w0.copy()
    n = max(len(y), 1)
    reg = 2 * lam / n
    def objective(w_):
        z = X @ w_
        # weighted mean log-loss, computed stably: log(1 + e^z) - y z
        return float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)) / n + reg / 2 * np.sum((w_ - w0) ** 2))

    obj = objective(w)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))
        grad = X.T @ (sw * (p - y)) / n + reg * (w - w0)
        hess = (X * (sw * p * (1 - p))[:, None]).T @ X / n + reg * np.eye(len(w))
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:  # numerically singular: minimum-norm step
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        # damped Newton: halve the step until the objective decreases (guaranteed convergence)
        t = 1.0
        while t > 1e-10:
            new_w = w - t * step
            new_obj = objective(new_w)
            if new_obj <= obj:
                break
            t /= 2
        else:
            break
        w, obj = new_w, new_obj
        if np.max(np.abs(t * step)) < tol:
            break
    return {k: float(v) for k, v in zip(keys, w)}


def predict(feature_dicts, weights, features):
    X = design_matrix(feature_dicts, features)
    w = np.array([weights["bias"]] + [weights[k] for k in features])
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))


def log_loss(feature_dicts, labels, weights, features):
    p = np.clip(predict(feature_dicts, weights, features), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
