from __future__ import annotations

import numpy as np


def score_matrix(actual: np.ndarray, prediction: np.ndarray, *, fit_mean_abs: float, peak_threshold: float, ramp_threshold: float) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.size == 0:
        raise ValueError("actual/prediction must be aligned [origin,H] arrays")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("actual or predicted values contain non-finite entries")
    yf, pf = y.reshape(-1), p.reshape(-1)
    error = pf - yf
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denom) if denom > 0 else float("nan")
    center = float(np.mean(yf))
    ia_denom = float(np.sum((np.abs(pf - center) + np.abs(yf - center)) ** 2))
    ia = float(1.0 - np.sum(error ** 2) / ia_denom) if ia_denom > 0 else float("nan")
    peak_mask = y >= peak_threshold
    peak_mae = float(np.mean(np.abs(y[peak_mask] - p[peak_mask]))) if peak_mask.any() else float("nan")
    ramp = np.abs(np.diff(y, axis=1))
    ramp_error = np.abs(np.diff(p, axis=1) - np.diff(y, axis=1))
    ramp_mask = ramp >= ramp_threshold
    ramp_mae = float(np.mean(ramp_error[ramp_mask])) if ramp_mask.any() else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "VAE": float(np.var(np.abs(error), ddof=0)),
        "IA": ia,
        "nMAE": mae / float(fit_mean_abs),
        "PeakMAE": peak_mae,
        "RampMAE": ramp_mae,
    }


def health_stats(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    target_std = float(np.std(y, ddof=0))
    prediction_std = float(np.std(p, ddof=0))
    if target_std > 0 and prediction_std > 0:
        correlation = float(np.corrcoef(y, p)[0, 1])
    else:
        correlation = float("nan")
    return {
        "target_mean": float(np.mean(y)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(p)),
        "prediction_std": prediction_std,
        "VR": prediction_std / target_std if target_std > 0 else float("nan"),
        "Pearson": correlation,
    }


def fit_thresholds(values: np.ndarray, quantile: float = 0.90) -> dict[str, float]:
    y = np.asarray(values, dtype=float).reshape(-1)
    return {
        "peak": float(np.quantile(y, quantile)),
        "ramp": float(np.quantile(np.abs(np.diff(y)), quantile)),
        "quantile": float(quantile),
    }
