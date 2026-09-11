from pathlib import Path

import numpy as np
import pandas as pd

from acharge import DATASETS, fit_acharge, load_dataset
from acharge.ensemble.fusion import fit_fusion, predict_fusion


def test_bundled_datasets_load_with_common_schema():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    for name in DATASETS:
        frame = load_dataset(name, data_dir)
        assert {"timestamp", "target", "calendar_hour", "calendar_dow"}.issubset(frame.columns)
        assert len(frame) > 0
        assert np.isfinite(frame["target"]).all()


def test_stability_fusion_round_trip():
    rng = np.random.default_rng(7)
    actual = rng.uniform(5.0, 15.0, size=(80, 24))
    predictions = np.stack([actual + rng.normal(0.0, scale, actual.shape) for scale in (0.2, 0.4, 0.6)], axis=1)
    regime = rng.normal(size=(80, 6))
    state = fit_fusion("SB", predictions, actual, regime)
    forecast, active = predict_fusion(state, predictions, regime)
    assert forecast.shape == actual.shape
    assert active.shape == (len(actual),)


def test_acharge_fit_and_predict(tmp_path):
    rng = np.random.default_rng(11)
    steps = np.arange(900)
    target_series = 20.0 + 3.0 * np.sin(2 * np.pi * steps / 24.0)
    frame = pd.DataFrame(
        {
            "target": target_series,
            "calendar_hour": steps % 24,
            "calendar_dow": (steps // 24) % 7,
            "calendar_weekend": ((steps // 24) % 7 >= 5).astype(int),
            "calendar_month": 1 + (steps // (24 * 30)) % 12,
            "availability_state": 1,
        }
    )
    history, horizon = 24, 6
    origins = np.arange(100, 500)
    actual = np.stack([target_series[o : o + horizon] for o in origins])
    predictions = np.stack(
        [actual + rng.normal(0.0, scale, actual.shape) for scale in (0.15, 0.25, 0.4, 0.7)],
        axis=1,
    )
    state = fit_acharge(
        frame=frame,
        origins=origins,
        endpoint_predictions=predictions,
        actual=actual,
        candidate_keys=("e1", "e2", "e3", "e4"),
        retention_r=(0.02, 0.03, 0.05, 0.08),
        retention_v=(0.01, 0.01, 0.02, 0.03),
        history=history,
        fit_mean_abs=float(np.mean(target_series[:100])),
        purge=history,
    )
    forecast, active = state.predict(frame, origins[:12], predictions[:12])
    assert forecast.shape == (12, horizon)
    assert active.shape == (12,)
    state_path = tmp_path / "state.json"
    state.save(state_path)
    restored = type(state).load(state_path)
    restored_forecast, restored_active = restored.predict(frame, origins[:12], predictions[:12])
    np.testing.assert_allclose(restored_forecast, forecast)
    np.testing.assert_allclose(restored_active, active)
