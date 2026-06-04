# 03 — Evaluation protocol

## Train/test split

- **Strategy**: temporal hold-out — the last 12 months of each series are held out as the test set.
- **Same cutoff per series**: every model predicts exactly the same out-of-sample window per `(institution, publisher)`.
- **No rolling cross-validation** is used in the main comparison (it is used inside PyCaret for best-model selection).
- **Fixed horizon**: `h = 12` months (a full year).

## Evaluation window

```
train: months [0, T-12)
test:  months [T-12, T)
```

where `T` is the last observed month for that series (it can vary by series).

## Standard metrics (the six)

All are computed **per series** and then reported as the median, the 25th/75th percentiles, and the range, by cluster or by publisher.

### MASE — Mean Absolute Scaled Error (primary)

Scales the model's MAE against the in-sample MAE of the Seasonal Naive forecast on the training set:

```
MASE = MAE_model / MAE_seasonal_naive_in_sample(series_train, s=12)
```

- `MASE < 1` → better than seasonal naive.
- `MASE = 1` → equivalent to seasonal naive.
- `MASE > 1` → worse than seasonal naive.
- Scale-free (comparable across series).

### MAE — Mean Absolute Error

```
MAE = mean(|y_true - y_pred|)
```

Error in real units (downloads/month). Useful for operational reporting.

### RMSE — Root Mean Squared Error

```
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Penalizes large errors. Sensitive to spikes.

### sMAPE — Symmetric Mean Absolute Percentage Error

```
sMAPE = mean( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) ) * 100
```

Range 0–200%. Robust to zeros (does not diverge when `y_true = 0`).

### MedAE — Median Absolute Error

```
MedAE = median(|y_true - y_pred|)
```

Robust to outliers within the horizon.

### MAPE — Mean Absolute Percentage Error

```
MAPE = mean(|y_true - y_pred| / y_true) * 100
```

**Computed only when `y_true > 0` in every test month.** If zeros are present it is reported with a warning (invalid denominator).

## Reporting rules

1. **MASE is the primary metric.** All model comparisons are ordered by MASE.
2. **The 6 metrics are always reported**, in a single table per (series × model).
3. **Summaries by cluster or by publisher** use the median, not the mean (the distributions are skewed with a long tail).
4. **Series with `train < 24 months`** are excluded from comparisons to keep a valid seasonal baseline.
5. **Series whose test values are all 0** are excluded from the MAPE computation but reported in the others.

## Mandatory baselines

Every comparison includes at least:
- **Seasonal Naive** (s=12) — defines the MASE denominator.
- Optionally: **Seasonal Mean** (mean of the corresponding months from previous years).

## Reproducibility

- Fixed seed (`random_state=42`) where applicable (PyCaret, KMeans).
- Package versions pinned in `requirements*.txt`.
- All foundation models are loaded from HuggingFace in deterministic mode.

## Reference implementation

See [src/metrics.py](../src/metrics.py) for the canonical implementation of the 6 metrics. All notebooks must import from there rather than reimplement them.
