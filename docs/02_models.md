# 02 — Models evaluated

Five models cover the comparison space: three modern foundation models, one classical AutoML, and one baseline.

---

## 1. TimesFM 2.5 (Google)

**Type:** Foundation, decoder-only, pre-trained on ~100B time points.

**Checkpoint:** `google/timesfm-2.5-200m-pytorch` (200M parameters, PyTorch backbone).

**Loading:**

```python
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

model = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True,
    return_backcast=False,
))
```

**Input:** a list of `np.ndarray[float32]` arrays, one series per element.

**Univariate inference:**

```python
point_forecasts, quantiles = model.forecast(
    horizon=12,
    inputs=series_list,
)
```

**Inference with covariates** (requires recompiling with `return_backcast=True`):

```python
point_forecasts, quantiles = model.forecast_with_covariates(
    horizon=12,
    inputs=series_list,
    dynamic_numerical_covariates=dynamic_covs,   # array [n_series x n_timesteps]
    static_numerical_covariates=static_covs,     # array [n_series x n_features]
)
```

**Quantiles returned:** 0.1 to 0.9 in steps of 0.1, with `fix_quantile_crossing=True` to avoid inversions.

**Notes:**
- `normalize_inputs=True` handles per-series scaling internally.
- The effective context is automatically adjusted to the available history (no padding).
- The backbone can be frozen for parametric fine-tuning (not used in this work).

---

## 2. Chronos-Bolt (Amazon)

**Type:** Foundation, encoder-decoder, based on T5, with quantile-based tokenization.

**Checkpoint:** `amazon/chronos-bolt-small`.

**Loading:**

```python
from chronos import BaseChronosPipeline
import torch

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
```

**Input:** a `torch.Tensor` of shape `(batch, context_length)`, dtype float32.

**Inference:**

```python
forecast = pipeline.predict(
    context=series_tensor,
    prediction_length=12,
)
# forecast.shape = (batch, num_quantiles, horizon)
median = forecast[:, forecast.shape[1] // 2, :]
```

**Standard post-processing:** `np.clip(predictions, 0, None)` to avoid negatives.

**Notes:**
- Chronos-Bolt is ~250× faster than classical Chronos while preserving competitive accuracy.
- It does not support exogenous covariates in its current API.
- It works well on series with a short context (≥24 points).

---

## 3. TabPFN-TS (Prior Labs)

**Type:** Tabular foundation model adapted to time series through automatic featurization.

**Package:** `tabpfn-time-series`.

**Loading:**

```python
from tabpfn_time_series import (
    TabPFNTimeSeriesPredictor,
    TabPFNMode,
    TimeSeriesDataFrame,
    FeatureTransformer,
)

predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
```

**Input:** a `TimeSeriesDataFrame` with a `MultiIndex(item_id, timestamp)` and a `target` column.

**Inference:**

```python
train_tsdf = TimeSeriesDataFrame.from_data_frame(
    df_train, id_column="item_id", timestamp_column="date",
)
test_tsdf = TimeSeriesDataFrame.from_data_frame(
    df_test_skeleton, id_column="item_id", timestamp_column="date",
)

pred_tsdf = predictor.predict(train_tsdf, test_tsdf)
```

**Automatic featurization:**
- `RunningIndexFeature`
- `CalendarFeature` (month, year)
- `AutoSeasonalFeature` (dominant-period detection)

**Covariates:** optional — added as extra columns to the `TimeSeriesDataFrame`.

**Post-processing:** the median column is extracted from the resulting `TimeSeriesDataFrame`.

---

## 4. PyCaret AutoML (classical)

**Type:** AutoML for time series. It trains ~30 classical models (ARIMA, ETS, Theta, Prophet, lag-based regressors, etc.) and selects the best one via cross-validation on temporal blocks.

**Package:** `pycaret[time-series]`.

**Per-series usage:**

```python
from pycaret.time_series import TSForecastingExperiment

exp = TSForecastingExperiment()
exp.setup(data=series_train, fh=12, seasonal_period=12, session_id=42, verbose=False)
best = exp.compare_models()
exp.finalize_model(best)
preds = exp.predict_model(best, fh=12)
```

**Notes:**
- Requires at least 36 months of history for the models with annual seasonality.
- It is the slowest per series (orders of magnitude more than any foundation model).
- The best model varies by series, which is why it is reported as `PyCaret_best`.

---

## 5. Seasonal Naive (baseline)

**Definition:** the forecast for month `t + h` is the value observed at `t + h − s`, where `s = 12`.

```python
def seasonal_naive(series, horizon, period=12):
    return series[-period:][:horizon]
```

**Use:** the MASE denominator and the minimal conceptual reference. Any model that does not beat Seasonal Naive on MASE adds no value over seasonality alone.

---

## Comparative summary

| Model | Parameters | Covariates | Typical time/series (GPU) | Minimum history |
|---|---|---|---|---|
| TimesFM 2.5 | 200M | Yes (xreg) | ~1–2 s | 12 months |
| Chronos-Bolt small | ~50M | No | ~0.2–0.4 s | 24 months |
| TabPFN-TS | varies | Yes (optional) | ~0.5–0.8 s | 24 months |
| PyCaret | varies | No | ~30–200 s | 36 months |
| Seasonal Naive | 0 | No | < 1 ms | 12 months |
