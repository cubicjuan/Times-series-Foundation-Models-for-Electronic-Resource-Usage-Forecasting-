# 06 — Series clustering and recommendation system

This section documents two blocks that complement the forecasting:

1. **Clustering** of series by usage profile (aggregated variables).
2. A **recommendation system** that combines forecast + cluster + disciplinary profile to produce an operational recommendation per series.

---

## Part A — Series clustering

### Objective

Segment the `(institution × publisher)` series into homogeneous behavioral profiles in order to:
- Compare models within each segment.
- Apply differentiated decision rules in the recommendation system.

### Per-series features

Each series is summarized into a feature vector computed over the last 60 observed months:

| Feature | Definition |
|---|---|
| `mean_volume` | Monthly average of downloads |
| `trend_slope` | Slope of a linear regression on the series |
| `cv` | Coefficient of variation (`std/mean`) |
| `seasonality_strength` | `var(monthly_means) / var(series)` |
| `zero_prop` | Proportion of months with value = 0 |
| `recent_ratio` | `mean(last 12m) / mean(first 24m)` (capped at 3.0) |

Additionally (when forecasts are available):

| Derived feature | Definition |
|---|---|
| `tabpfn_slope` | Slope of the 12-month forecast |
| `tabpfn_ratio` | `sum(forecast next 12m) / sum(actual last 12m)` |
| `volume_norm_enrollment` | `mean_volume / institutional_enrollment` |

### Algorithm

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = features_df.values
X_scaled = StandardScaler().fit_transform(X)

# K selection by silhouette
for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    s = silhouette_score(X_scaled, labels)
    print(k, s)

# Final K fixed by silhouette analysis + interpretability
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10).fit(X_scaled)
```

### Cluster labeling

Labels are assigned **by interpreting the centroids** (not by numeric order):

| Label | Centroid characteristics |
|---|---|
| `A_consolidated` | High volume, flat or slight trend, low variability |
| `B_growth` | Strong positive trend, `recent_ratio > 1.5` |
| `C_intermittent` | High `zero_prop` or extreme `cv` |
| `D_decline` | Negative trend, `recent_ratio < 1` |

The final number of clusters depends on the silhouette analysis; typically K=2 or K=4.

### Standard outputs

| File | Content |
|---|---|
| `series_profiles.csv` | `series_id, cluster_id, cluster_label, features...` |
| `cluster_centroids.csv` | Centroids in the original space |
| `cluster_vs_model_mase.csv` | Median MASE per (cluster × model) |
| `cluster_vs_cov_impact.csv` | Δ MASE of adding covariates, per cluster |

---

## Part B — Recommendation system

### Objective

Produce, for each series, an operational recommendation for the next year, in one of three categories:

| Recommendation | Meaning |
|---|---|
| `validated_investment` | Solid or growing usage, **renew**. |
| `renegotiation` | Strong projected drop or poor performance, **renegotiate or consider cancellation**. |
| `promotion_deepening` | Intermediate or low-confidence case, **keep active but prioritize promotion**. |

### Inputs

1. Forecasts from the model selected per series (see selection logic below).
2. Actual usage of the last observed year.
3. Cluster label (from Part A).
4. Disciplinary-relevance score: `score = Σ (enrollment_area_i × publisher_catalog_area_i)`.
5. Dominant SJR quartile of the publisher.
6. Historical-confidence metric (median MASE of the model in validation).

### Decision logic

```python
def recommend(series, pred_2026, actual_2025, cluster, sjr, relevance, confidence):
    change_pct = (pred_2026 - actual_2025) / actual_2025 * 100

    # Thresholds differentiated by cluster
    threshold_inv = -10 if cluster == "A_consolidated" else -5
    threshold_reneg = -30 if cluster == "A_consolidated" else -20

    # SJR adjustments
    if sjr == "Q1":
        threshold_reneg -= 10   # more permissive (Q1 is worth retaining)
    elif sjr == "Q4":
        threshold_reneg += 10   # stricter

    # Adjustment for low disciplinary relevance
    if relevance < 0.10:
        threshold_reneg += 5

    if confidence in ("low", "no_data"):
        return "promotion_deepening"
    if change_pct >= threshold_inv:
        return "validated_investment"
    if change_pct < threshold_reneg:
        return "renegotiation"
    return "promotion_deepening"
```

### Per-series forecast-model selection

```python
def select_model(series):
    delta_cov = MASE_with_cov - MASE_without_cov
    if delta_cov < -0.01:  # covariates help significantly
        return "TimesFM_with_covariates"
    return model_with_lowest_historical_MASE(series)
```

### Flagging unreliable forecasts

Series with `|change_pct| > 80%` or forecasts > 200% of the historical level are flagged as `confidence='no_data'` and routed to `promotion_deepening` by default.

### Outputs

| File | Content |
|---|---|
| `annual_recommendations.csv` | One row per series with: `cluster_label`, `actual_usage`, `predicted_usage`, `change_pct`, `recommendation`, `confidence`, `relevance_score` |
| `model_selection_rule.csv` | Model chosen per series + justification |

---

## Implementation

Reference notebooks:
- [notebooks/07_clustering_series.ipynb](../notebooks/07_clustering_series.ipynb)
- [notebooks/08_recommendation_system.ipynb](../notebooks/08_recommendation_system.ipynb)
