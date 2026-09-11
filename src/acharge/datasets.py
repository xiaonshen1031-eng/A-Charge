from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "suzhou": {
        "file": "suzhou_hourly_load.parquet",
        "target": "load_kw",
        "label": "Suzhou regional charging load",
    },
    "zhenjiang": {
        "file": "zhenjiang_hourly_load.parquet",
        "target": "load_kw",
        "label": "Zhenjiang regional charging load",
    },
    "shenzhen": {
        "file": "shenzhen_hourly_occupancy.parquet",
        "target": "occupied_piles",
        "label": "Shenzhen occupied charging piles",
    },
    "los_angeles": {
        "file": "los_angeles_hourly_charging.parquet",
        "target": "target_kw",
        "label": "Los Angeles city charging load",
    },
}


def default_data_dir() -> Path:
    configured = os.environ.get("ACHARGE_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def load_dataset(name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load one bundled city series and expose a common hourly target schema."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in DATASETS:
        choices = ", ".join(DATASETS)
        raise KeyError(f"unknown dataset {name!r}; choose one of: {choices}")
    spec = DATASETS[key]
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    frame = pd.read_parquet(root / spec["file"])
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["timestamp"] = timestamp
    frame["target"] = pd.to_numeric(frame[spec["target"]], errors="raise").astype(float)
    frame["calendar_hour"] = timestamp.dt.hour.astype(np.int16)
    frame["calendar_dow"] = timestamp.dt.dayofweek.astype(np.int16)
    frame["calendar_weekend"] = (timestamp.dt.dayofweek >= 5).astype(np.int8)
    frame["calendar_month"] = timestamp.dt.month.astype(np.int16)
    frame["availability_state"] = np.ones(len(frame), dtype=np.int8)
    return frame
