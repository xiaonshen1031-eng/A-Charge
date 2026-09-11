from __future__ import annotations

from typing import Any

import numpy as np


FUSION_CANDIDATES = ("RSF", "LAR", "NAR", "SB", "HB")


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    denominator = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, denominator, out=np.full_like(weights, 1.0 / weights.shape[-1]), where=denominator > 0)


def fit_fusion(name: str, predictions: np.ndarray, actual: np.ndarray, regime: np.ndarray) -> dict[str, Any]:
    """Fit one A-Charge fusion candidate on CAL-train only.

    Predictions are [origin, expert, lead], actual is [origin, lead].
    Returned states contain only portable numeric parameters.
    """
    pred = np.asarray(predictions, dtype=float)
    y = np.asarray(actual, dtype=float)
    x = np.asarray(regime, dtype=float)
    if pred.ndim != 3 or y.shape != (pred.shape[0], pred.shape[2]) or x.shape[0] != len(pred):
        raise ValueError("misaligned fusion training arrays")
    if not (np.isfinite(pred).all() and np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("fusion training contains non-finite values")
    error = np.abs(pred - y[:, None, :])
    if name == "LAR":
        weights = _normalize(1.0 / (error.mean(axis=0).T + 1e-8))  # [lead, expert]
        return {"name": name, "weights": weights.tolist()}
    if name == "SB":
        level = error.mean(axis=(0, 2))
        instability = error.mean(axis=2).std(axis=0)
        weights = _normalize(1.0 / (level + instability + 1e-8))
        return {"name": name, "weights": weights.tolist()}
    if name == "HB":
        from scipy.optimize import nnls
        design = np.transpose(pred, (0, 2, 1)).reshape(-1, pred.shape[1])
        weights, _ = nnls(design, y.reshape(-1))
        weights = _normalize(weights)
        return {"name": name, "weights": weights.tolist()}
    if name == "RSF":
        volatility_column = min(4, x.shape[1] - 1)
        thresholds = np.quantile(x[:, volatility_column], [1.0 / 3.0, 2.0 / 3.0])
        labels = np.digitize(x[:, volatility_column], thresholds)
        weights = []
        global_weights = _normalize(1.0 / (error.mean(axis=(0, 2)) + 1e-8))
        for group in range(3):
            mask = labels == group
            group_weights = _normalize(1.0 / (error[mask].mean(axis=(0, 2)) + 1e-8)) if mask.any() else global_weights
            weights.append(group_weights)
        return {"name": name, "volatility_column": volatility_column, "thresholds": thresholds.tolist(), "weights": np.asarray(weights).tolist()}
    if name == "NAR":
        from sklearn.linear_model import LogisticRegression
        labels = np.argmin(error.mean(axis=2), axis=1)
        classes = np.unique(labels)
        if len(classes) == 1:
            return {"name": name, "constant_expert": int(classes[0]), "expert_count": int(pred.shape[1])}
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        model.fit(x, labels)
        return {
            "name": name,
            "classes": model.classes_.astype(int).tolist(),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "expert_count": int(pred.shape[1]),
        }
    raise KeyError(name)


def predict_fusion(state: dict[str, Any], predictions: np.ndarray, regime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predictions, dtype=float)
    x = np.asarray(regime, dtype=float)
    name = str(state["name"])
    if pred.ndim != 3 or x.shape[0] != len(pred):
        raise ValueError("misaligned fusion prediction arrays")
    n, experts, horizon = pred.shape
    if name == "LAR":
        weights = np.asarray(state["weights"], dtype=float)  # [lead, expert]
        output = np.einsum("nel,le->nl", pred, weights)
        active = np.full(n, np.sum(weights > 1e-12, axis=1).mean())
    elif name in ("SB", "HB"):
        weights = np.asarray(state["weights"], dtype=float)
        output = np.einsum("nel,e->nl", pred, weights)
        active = np.full(n, np.sum(weights > 1e-12))
    elif name == "RSF":
        column = int(state["volatility_column"])
        labels = np.digitize(x[:, column], np.asarray(state["thresholds"], dtype=float))
        weights = np.asarray(state["weights"], dtype=float)[labels]
        output = np.einsum("nel,ne->nl", pred, weights)
        active = np.sum(weights > 1e-12, axis=1)
    elif name == "NAR":
        weights = np.zeros((n, experts), dtype=float)
        if "constant_expert" in state:
            weights[:, int(state["constant_expert"])] = 1.0
        else:
            coef = np.asarray(state["coef"], dtype=float)
            intercept = np.asarray(state["intercept"], dtype=float)
            logits = x @ coef.T + intercept
            if logits.shape[1] == 1 and len(state["classes"]) == 2:
                logits = np.column_stack([np.zeros(n), logits[:, 0]])
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
            weights[:, np.asarray(state["classes"], dtype=int)] = probability
        output = np.einsum("nel,ne->nl", pred, weights)
        active = np.sum(weights > 1e-6, axis=1)
    else:
        raise KeyError(name)
    return np.maximum(output, 0.0), active.astype(float)
