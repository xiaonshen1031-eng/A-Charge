from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REGIME_FEATURES = (
    "last_load", "mean_24", "mean_history", "normalized_level",
    "std_24", "mad_history", "last_abs_ramp", "mean_abs_ramp_24",
    "daily_displacement", "weekly_or_history_displacement",
    "calendar_hour_sin", "calendar_hour_cos", "calendar_dow_sin", "calendar_dow_cos",
    "calendar_weekend", "calendar_month_sin", "calendar_month_cos",
    "expert_disagreement_std", "expert_disagreement_iqr", "expert_disagreement_spread",
    "availability_state",
)


def build_issue_time_features(
    frame: pd.DataFrame,
    origins: np.ndarray,
    expert_predictions: np.ndarray,
    *,
    history: int,
    fit_mean_abs: float,
) -> pd.DataFrame:
    origins = np.asarray(origins, dtype=int)
    predictions = np.asarray(expert_predictions, dtype=float)
    if predictions.ndim != 3 or predictions.shape[0] != len(origins):
        raise ValueError("expert predictions must be [origin,expert,lead]")
    if not np.isfinite(predictions).all() or fit_mean_abs <= 0:
        raise ValueError("finite predictions and positive FIT scale required")
    rows = []
    for row_index, origin in enumerate(origins):
        if origin < history or origin >= len(frame):
            raise ValueError("origin lacks causal history")
        target = frame.iloc[origin-history:origin]["target"].to_numpy(dtype=float)
        current = frame.iloc[origin]
        ramp = np.abs(np.diff(target))
        prediction = predictions[row_index]
        q75, q25 = np.quantile(prediction, [0.75, 0.25], axis=0)
        daily_reference = target[-25] if len(target) >= 25 else target[0]
        seasonal_reference = target[-168] if len(target) >= 168 else target[0]
        rows.append({
            "last_load": float(target[-1]),
            "mean_24": float(np.mean(target[-min(24, len(target)):])),
            "mean_history": float(np.mean(target)),
            "normalized_level": float(target[-1] / fit_mean_abs),
            "std_24": float(np.std(target[-min(24, len(target)):])),
            "mad_history": float(np.median(np.abs(target - np.median(target)))),
            "last_abs_ramp": float(ramp[-1]) if len(ramp) else 0.0,
            "mean_abs_ramp_24": float(np.mean(ramp[-min(24, len(ramp)):])) if len(ramp) else 0.0,
            "daily_displacement": float(target[-1] - daily_reference),
            "weekly_or_history_displacement": float(target[-1] - seasonal_reference),
            "calendar_hour_sin": float(np.sin(2 * np.pi * float(current["calendar_hour"]) / 24.0)),
            "calendar_hour_cos": float(np.cos(2 * np.pi * float(current["calendar_hour"]) / 24.0)),
            "calendar_dow_sin": float(np.sin(2 * np.pi * float(current["calendar_dow"]) / 7.0)),
            "calendar_dow_cos": float(np.cos(2 * np.pi * float(current["calendar_dow"]) / 7.0)),
            "calendar_weekend": float(current["calendar_weekend"]),
            "calendar_month_sin": float(np.sin(2 * np.pi * (float(current["calendar_month"]) - 1) / 12.0)),
            "calendar_month_cos": float(np.cos(2 * np.pi * (float(current["calendar_month"]) - 1) / 12.0)),
            "expert_disagreement_std": float(np.mean(np.std(prediction, axis=0))),
            "expert_disagreement_iqr": float(np.mean(q75 - q25)),
            "expert_disagreement_spread": float(np.mean(np.max(prediction, axis=0) - np.min(prediction, axis=0)) / fit_mean_abs),
            "availability_state": float(current.get("availability_state", 1.0)),
        })
    result = pd.DataFrame(rows, columns=REGIME_FEATURES)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("non-finite regime features")
    return result


@dataclass
class RegimeTransformer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: pd.DataFrame) -> "RegimeTransformer":
        values = features.loc[:, REGIME_FEATURES].to_numpy(dtype=float)
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("transformer is not fitted")
        return (features.loc[:, REGIME_FEATURES].to_numpy(dtype=float) - self.mean_) / self.scale_

    def to_dict(self) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("transformer is not fitted")
        return {"feature_names": list(REGIME_FEATURES), "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegimeTransformer":
        if tuple(payload["feature_names"]) != REGIME_FEATURES:
            raise ValueError("regime feature schema mismatch")
        return cls(np.asarray(payload["mean"], dtype=float), np.asarray(payload["scale"], dtype=float))
