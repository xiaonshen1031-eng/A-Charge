from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ensemble.complementarity import select_portfolio
from .ensemble.fusion import FUSION_CANDIDATES, fit_fusion, predict_fusion
from .ensemble.selection import select_fusion_candidate
from .metrics import fit_thresholds, score_matrix
from .regime import RegimeTransformer, build_issue_time_features


def cal_forward_folds(origins: np.ndarray, *, purge: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    origin = np.asarray(origins, dtype=int)
    if len(origin) < 3 * purge or np.any(np.diff(origin) <= 0):
        raise ValueError("CAL origins must be ordered and long enough for two forward folds")
    cut1, cut2 = len(origin) // 3, (2 * len(origin)) // 3
    folds = []
    for train_stop, validation_stop in ((cut1, cut2), (cut2, len(origin))):
        validation_origin = origin[train_stop - 1] + purge + 1
        validation_start = int(np.searchsorted(origin, validation_origin, side="left"))
        train = np.arange(0, train_stop, dtype=int)
        validation = np.arange(validation_start, validation_stop, dtype=int)
        if not len(train) or not len(validation):
            raise ValueError("purge interval removed a calibration fold")
        folds.append({"train": train, "validation": validation})
    return tuple(folds)  # type: ignore[return-value]


def _candidate_manifest(
    actual: np.ndarray,
    predictions: np.ndarray,
    candidate_keys: Sequence[str],
    retention_r: Sequence[float],
    retention_v: Sequence[float],
    *,
    q: float,
    fit_mean_abs: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = actual[:, None, :] - predictions
    flat = residual.transpose(0, 2, 1).reshape(-1, predictions.shape[1])
    correlation = pd.DataFrame(
        np.corrcoef(flat, rowvar=False),
        index=list(candidate_keys),
        columns=list(candidate_keys),
    )
    if not np.isfinite(correlation.to_numpy()).all():
        raise ValueError("candidate residual correlation contains non-finite values")
    rows = []
    for index, key in enumerate(candidate_keys):
        absolute = np.abs(residual[:, index, :]).reshape(-1)
        threshold = float(np.quantile(absolute, q))
        tail = float(np.mean(absolute[absolute >= threshold])) / fit_mean_abs
        rows.append(
            {
                "candidate_key": key,
                "retention_R": float(retention_r[index]),
                "retention_T": tail,
                "retention_V": float(retention_v[index]),
            }
        )
    return pd.DataFrame(rows), correlation


def _crossfit_fusion(
    name: str,
    origins: np.ndarray,
    predictions: np.ndarray,
    actual: np.ndarray,
    raw_features: pd.DataFrame,
    *,
    fit_mean_abs: float,
    q: float,
    purge: int,
) -> dict[str, Any]:
    oof = np.full_like(actual, np.nan, dtype=float)
    fold_rows = []
    active = np.full(len(actual), np.nan, dtype=float)
    for fold_number, fold in enumerate(cal_forward_folds(origins, purge=purge), start=1):
        train, validation = fold["train"], fold["validation"]
        transformer = RegimeTransformer().fit(raw_features.iloc[train])
        state = fit_fusion(
            name,
            predictions[train],
            actual[train],
            transformer.transform(raw_features.iloc[train]),
        )
        forecast, counts = predict_fusion(
            state,
            predictions[validation],
            transformer.transform(raw_features.iloc[validation]),
        )
        oof[validation] = forecast
        active[validation] = counts
        thresholds = fit_thresholds(actual[train].reshape(-1), quantile=q)
        metrics = score_matrix(
            actual[validation],
            forecast,
            fit_mean_abs=fit_mean_abs,
            peak_threshold=thresholds["peak"],
            ramp_threshold=thresholds["ramp"],
        )
        peak_mask = actual[validation] >= thresholds["peak"]
        ramp_mask = np.abs(np.diff(actual[validation], axis=1)) >= thresholds["ramp"]
        peak_nmae = float(metrics["PeakMAE"]) / fit_mean_abs if peak_mask.any() else float(metrics["nMAE"])
        ramp_nmae = float(metrics["RampMAE"]) / fit_mean_abs if ramp_mask.any() else float(metrics["nMAE"])
        risk = float(metrics["nMAE"] + 0.25 * peak_nmae + 0.25 * ramp_nmae)
        fold_rows.append(
            {
                "fold": fold_number,
                "R_cal": risk,
                "metrics": metrics,
                "Peak_nMAE": peak_nmae,
                "Ramp_nMAE": ramp_nmae,
                "active_expert_count": float(np.mean(counts)),
            }
        )
    evaluated = np.isfinite(oof).all(axis=1)
    per_origin_loss = np.mean(np.abs(actual[evaluated] - oof[evaluated]), axis=1) / fit_mean_abs
    return {
        "name": name,
        "folds": fold_rows,
        "macro_R_cal": float(np.mean([row["R_cal"] for row in fold_rows])),
        "macro_nMAE": float(np.mean([row["metrics"]["nMAE"] for row in fold_rows])),
        "per_origin_loss": per_origin_loss,
        "active_expert_count": float(np.nanmean(active)),
    }


@dataclass(frozen=True)
class AChargeState:
    candidate_keys: tuple[str, ...]
    selected_indices: tuple[int, ...]
    history: int
    fit_mean_abs: float
    portfolio: dict[str, Any]
    fusion_selection: dict[str, Any]
    regime_transform: dict[str, Any]
    fusion_state: dict[str, Any]

    def predict(
        self,
        frame: pd.DataFrame,
        origins: np.ndarray,
        endpoint_predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.asarray(endpoint_predictions, dtype=float)
        selected = predictions[:, self.selected_indices, :]
        raw = build_issue_time_features(
            frame,
            np.asarray(origins, dtype=int),
            selected,
            history=self.history,
            fit_mean_abs=self.fit_mean_abs,
        )
        transformer = RegimeTransformer.from_dict(self.regime_transform)
        return predict_fusion(self.fusion_state, selected, transformer.transform(raw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_keys": list(self.candidate_keys),
            "selected_indices": list(self.selected_indices),
            "history": self.history,
            "fit_mean_abs": self.fit_mean_abs,
            "portfolio": self.portfolio,
            "fusion_selection": self.fusion_selection,
            "regime_transform": self.regime_transform,
            "fusion_state": self.fusion_state,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AChargeState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["candidate_keys"] = tuple(payload["candidate_keys"])
        payload["selected_indices"] = tuple(payload["selected_indices"])
        return cls(**payload)


def fit_acharge(
    *,
    frame: pd.DataFrame,
    origins: np.ndarray,
    endpoint_predictions: np.ndarray,
    actual: np.ndarray,
    candidate_keys: Sequence[str],
    retention_r: Sequence[float],
    retention_v: Sequence[float],
    history: int,
    fit_mean_abs: float,
    k: int = 3,
    lambda_c: float = 0.25,
    q: float = 0.85,
    purge: int | None = None,
) -> AChargeState:
    """Fit A-Charge from calibration endpoint forecasts and issue-time states."""
    origin = np.asarray(origins, dtype=int)
    predictions = np.asarray(endpoint_predictions, dtype=float)
    target = np.asarray(actual, dtype=float)
    keys = tuple(str(key) for key in candidate_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("candidate keys must be unique")
    if predictions.ndim != 3 or target.shape != (len(origin), predictions.shape[2]):
        raise ValueError("endpoint predictions and actual values have incompatible shapes")
    if predictions.shape[1] != len(keys) or len(retention_r) != len(keys) or len(retention_v) != len(keys):
        raise ValueError("candidate metadata does not match the endpoint axis")
    manifest, correlation = _candidate_manifest(
        target,
        predictions,
        keys,
        retention_r,
        retention_v,
        q=q,
        fit_mean_abs=float(fit_mean_abs),
    )
    portfolio = select_portfolio(manifest, correlation, k=k, lambda_c=lambda_c)
    index = {key: position for position, key in enumerate(keys)}
    selected_indices = tuple(index[key] for key in portfolio["selected_experts"])
    selected = predictions[:, selected_indices, :]
    raw = build_issue_time_features(
        frame,
        origin,
        selected,
        history=history,
        fit_mean_abs=float(fit_mean_abs),
    )
    purge_size = int(history if purge is None else purge)
    fusion_results = {
        name: _crossfit_fusion(
            name,
            origin,
            selected,
            target,
            raw,
            fit_mean_abs=float(fit_mean_abs),
            q=q,
            purge=purge_size,
        )
        for name in FUSION_CANDIDATES
    }
    block_length = 24 if history >= 168 else 6
    fusion_selection = select_fusion_candidate(fusion_results, block_length=block_length)
    transformer = RegimeTransformer().fit(raw)
    fusion_state = fit_fusion(
        fusion_selection["selected_fusion"],
        selected,
        target,
        transformer.transform(raw),
    )
    return AChargeState(
        candidate_keys=keys,
        selected_indices=selected_indices,
        history=int(history),
        fit_mean_abs=float(fit_mean_abs),
        portfolio=portfolio,
        fusion_selection=fusion_selection,
        regime_transform=transformer.to_dict(),
        fusion_state=fusion_state,
    )
