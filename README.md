# A-Charge

A-Charge selects a compact expert portfolio from calibration forecasts and combines it with adaptive fusion driven by load state, calendar state, and expert disagreement.

## Repository contents

- `src/acharge/calibration.py`: portfolio construction, forward calibration, fusion selection, state fitting, and prediction.
- `src/acharge/ensemble/`: complementarity objective, five fusion families, and fusion selection.
- `src/acharge/regime.py`: issue-time state features and feature transformation.
- `src/acharge/metrics.py`: calibration metrics and event thresholds.
- `src/acharge/datasets.py`: loaders for the four bundled processed datasets.
- `data/processed/`: Suzhou, Zhenjiang, Shenzhen, and Los Angeles hourly data.
- `requirements.txt`, `environment.yml`, and `pyproject.toml`: Python environment and package metadata.

## Environment

```bash
conda env create -f environment.yml
conda activate acharge
pip install -e .
```

The package requires Python 3.10 or later. The recorded environment uses Python 3.11.9.

Run the package checks with:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Processed datasets

| Key | File | Rows | Time range | Forecast target |
|---|---|---:|---|---|
| `suzhou` | `suzhou_hourly_load.parquet` | 26,280 | 2021-01-01 to 2023-12-31 | Regional charging load in kW |
| `zhenjiang` | `zhenjiang_hourly_load.parquet` | 17,520 | 2022-01-01 to 2023-12-31 | Regional charging load in kW |
| `shenzhen` | `shenzhen_hourly_occupancy.parquet` | 720 | 2022-06-19 to 2022-07-18 | Occupied charging-pile count |
| `los_angeles` | `los_angeles_hourly_charging.parquet` | 4,392 | 2023-04-01 to 2023-09-30 | City charging load in kW |

Load a series with the common `timestamp` and `target` columns:

```python
from acharge import load_dataset

frame = load_dataset("suzhou")
```

## A-Charge API

`fit_acharge` consumes calibration endpoint forecasts with shape `[origin, candidate, lead]`, matching targets with shape `[origin, lead]`, candidate risk and variability vectors, and the hourly frame used to build issue-time features.

```python
from acharge import fit_acharge

state = fit_acharge(
    frame=frame,
    origins=cal_origins,
    endpoint_predictions=cal_predictions,
    actual=cal_actual,
    candidate_keys=candidate_keys,
    retention_r=cal_risk,
    retention_v=fold_variability,
    history=168,
    fit_mean_abs=fit_mean_abs,
    k=3,
    lambda_c=0.25,
    q=0.85,
)

forecast, active_experts = state.predict(frame, test_origins, test_predictions)
state.save("acharge_state.json")
```
