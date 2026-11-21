"""
MAP logistic regression (numpy only), shared by the importance and ranking models.

    minimise  mean_i logloss(sigmoid(w · x_i), y_i)  +  (lam / n) * ||w - w0||^2

x_i = [1, f_1, ..., f_k] (bias first). w0 are prior weights: they keep fits
stable on small datasets and pull rarely-active features back to sensible
values instead of letting them drift. The objective is convex; full-batch
gradient descent converges to the unique optimum.
"""
import numpy as np


def design_matrix(feature_dicts, features):
    return np.array([[1.0] + [float(f.get(k, 0.0)) for k in features] for f in feature_dicts], dtype=float)


def fit(feature_dicts, labels, prior, features, lam=0.5, lr=0.1, iters=4000):
    """prior: dict with "bias" and every name in `features`. Returns a weight dict."""
    keys = ["bias"] + list(features)
    X = design_matrix(feature_dicts, features)
    y = np.asarray(labels, dtype=float)
    w0 = np.array([prior[k] for k in keys], dtype=float)
    w = w0.copy()
    n = max(len(y), 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        w -= lr * (X.T @ (p - y) / n + 2 * lam * (w - w0) / n)
    return {k: float(v) for k, v in zip(keys, w)}


def predict(feature_dicts, weights, features):
    X = design_matrix(feature_dicts, features)
    w = np.array([weights["bias"]] + [weights[k] for k in features])
    return 1.0 / (1.0 + np.exp(-(X @ w)))


def log_loss(feature_dicts, labels, weights, features):
    p = np.clip(predict(feature_dicts, weights, features), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
