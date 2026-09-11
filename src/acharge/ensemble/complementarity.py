from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["origin_index", "lead"]


def aligned_residual_correlation(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(predictions) < 2:
        raise ValueError("at least two experts required")
    residual_columns = []
    expected_keys = None
    expected_actual = None
    for candidate_key, frame in predictions.items():
        required = set(KEYS + ["actual", "prediction"])
        if missing := required.difference(frame.columns):
            raise ValueError(f"prediction artifact missing {sorted(missing)}")
        ordered = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
        if ordered.duplicated(KEYS).any():
            raise ValueError(f"duplicate residual keys for {candidate_key}")
        keys = ordered.loc[:, KEYS]
        actual = ordered["actual"].to_numpy(dtype=float)
        if expected_keys is None:
            expected_keys = keys
            expected_actual = actual
        elif not keys.equals(expected_keys) or not np.array_equal(actual, expected_actual):
            raise ValueError("candidate residual vectors are not aligned on identical CAL origins/leads")
        residual = actual - ordered["prediction"].to_numpy(dtype=float)
        if not np.isfinite(residual).all() or float(np.var(residual)) <= 0.0:
            raise ValueError(f"undefined complementarity evidence for {candidate_key}: residual variance is zero/nonfinite")
        residual_columns.append(pd.Series(residual, name=candidate_key))
    residual_frame = pd.concat(residual_columns, axis=1)
    correlation = residual_frame.corr(method="pearson")
    off_diagonal = correlation.to_numpy()[~np.eye(len(correlation), dtype=bool)]
    if not np.isfinite(off_diagonal).all():
        raise ValueError("residual correlation contains non-finite values")
    return correlation, residual_frame


def _mean_abs_off_diagonal(correlation: pd.DataFrame, keys: tuple[str, ...]) -> float:
    matrix = correlation.loc[list(keys), list(keys)].to_numpy(dtype=float)
    return float(np.mean(np.abs(matrix[np.triu_indices(len(keys), k=1)])))


def select_portfolio(
    candidate_manifest: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    k: int,
    lambda_c: float,
) -> dict[str, Any]:
    if k < 2 or k > len(candidate_manifest):
        raise ValueError("invalid portfolio size")
    indexed = candidate_manifest.set_index("candidate_key", drop=False)
    if set(indexed.index) != set(correlation.index):
        raise ValueError("manifest/correlation candidate mismatch")
    best: tuple[float, tuple[str, ...], dict[str, float]] | None = None
    keys = tuple(indexed.index.astype(str))
    risks = indexed["retention_R"].to_numpy(dtype=float)
    tails = indexed["retention_T"].to_numpy(dtype=float)
    variability = indexed["retention_V"].to_numpy(dtype=float)
    corr = correlation.loc[list(keys), list(keys)].to_numpy(dtype=float)
    pair_index = np.triu_indices(k, k=1)
    for subset_index in combinations(range(len(keys)), k):
        indices = np.fromiter(subset_index, dtype=int, count=k)
        subset = tuple(keys[index] for index in subset_index)
        subcorr = corr[np.ix_(indices, indices)]
        components = {
            "mean_R": float(np.mean(risks[indices])),
            "mean_abs_residual_correlation": float(np.mean(np.abs(subcorr[pair_index]))),
            "mean_T": float(np.mean(tails[indices])),
            "mean_V": float(np.mean(variability[indices])),
        }
        objective = (
            components["mean_R"]
            + float(lambda_c) * components["mean_abs_residual_correlation"]
            + 0.25 * components["mean_T"]
            + 0.10 * components["mean_V"]
        )
        candidate = (float(objective), subset, components)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    assert best is not None
    return {
        "K": int(k),
        "lambda_c": float(lambda_c),
        "selected_experts": list(best[1]),
        "J_port": best[0],
        **best[2],
    }
