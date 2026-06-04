"""The six standard evaluation metrics of the project.

Each metric takes aligned 1-D numpy arrays (y_true, y_pred) and returns a
float scalar. MASE additionally requires the training series in order to
compute the Seasonal Naive denominator.

Convention:
  - y_true, y_pred: shape (h,) with h = forecast horizon
  - train: shape (n,) with n = training-set length
  - period: defaults to 12 (annual monthly seasonality)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8
SEASONAL_PERIOD = 12


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes do not match: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    return y_true, y_pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE as a percentage (0-200%). Robust to zeros."""
    y_true, y_pred = _check(y_true, y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + EPS
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.median(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE as a percentage. Only valid when y_true > 0 at every point."""
    y_true, y_pred = _check(y_true, y_pred)
    if np.any(y_true <= 0):
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train: np.ndarray,
    period: int = SEASONAL_PERIOD,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Denominator = in-sample MAE of the Seasonal Naive forecast on the
    training series.
    """
    y_true, y_pred = _check(y_true, y_pred)
    train = np.asarray(train, dtype=np.float64)
    if len(train) <= period:
        raise ValueError(
            f"Train has {len(train)} points, insufficient for "
            f"Seasonal Naive with period={period}"
        )
    seasonal_diff = np.abs(train[period:] - train[:-period])
    mae_naive = seasonal_diff.mean()
    if mae_naive < EPS:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / mae_naive)


def all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train: np.ndarray,
    period: int = SEASONAL_PERIOD,
) -> dict[str, float]:
    """Return the six metrics as a dict."""
    return {
        "MASE": mase(y_true, y_pred, train, period),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MedAE": medae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def seasonal_naive(train: np.ndarray, horizon: int, period: int = SEASONAL_PERIOD) -> np.ndarray:
    """Seasonal Naive baseline forecast."""
    train = np.asarray(train, dtype=np.float64)
    if len(train) < period:
        raise ValueError(
            f"Train has {len(train)} points, insufficient for period={period}"
        )
    base = train[-period:]
    repetitions = int(np.ceil(horizon / period))
    extended = np.tile(base, repetitions)
    return extended[:horizon]
