# 05 — Results (aggregated, anonymized)

This document summarizes the findings of the work without identifiers. Numeric values are reported as medians and ranges aggregated by **cluster** and by **publisher category**, never by individual institution.

---

## Main findings

### 1. Foundation models beat the seasonal baseline

In every per-cluster comparison, the modern foundation models obtain a median MASE **lower than Seasonal Naive** (reference MASE = 1.0). PyCaret AutoML also beats the baseline, but at a much higher computational cost.

### 2. There is no single winner among the foundation models

Over the full set of series, the distribution of the "best model per series" is split between **TimesFM, Chronos, and TabPFN-TS** with no clear dominance. The optimal choice is **mixed by cluster and by publisher**.

Example distribution of winners (no identifiers):

| Model | % of series where it is best |
|---|---|
| Chronos-Bolt | ~29% |
| TimesFM + covariates | ~24% |
| TabPFN-TS + cov | ~19% |
| PyCaret AutoML | ~18% |
| TabPFN-TS (univariate) | ~11% |

### 3. Covariates do not always help

When adding exogenous covariates (institutional enrollment, publisher subject profile) to TimesFM or TabPFN-TS, **the improvement is marginal and not statistically significant** in most clusters (Wilcoxon `p > 0.5`). The univariate version of TabPFN-TS is competitive with the covariate version.

**Practical implication:** prefer the simpler (univariate) version unless there is evidence of consistent improvement in a specific segment.

### 4. Speed: Chronos-Bolt is orders of magnitude faster than PyCaret

Typical per-series inference times (consumer-grade GPU):

| Model | Time/series |
|---|---|
| Seasonal Naive | < 1 ms |
| Chronos-Bolt small | ~0.2–0.4 s |
| TabPFN-TS | ~0.5–0.8 s |
| TimesFM 2.5 | ~1–2 s |
| PyCaret AutoML | ~30–200 s |

PyCaret is ~200–500× slower than any foundation model, **with no consistent accuracy gain**.

### 5. Problematic series

A subset of series (~5–10% of the total) shows MASE > 2.8 across **all models**, indicating that the problem lies not in the model but in data quality:
- Unforeseeable structural changes (contract renegotiations).
- Short or intermittent temporal coverage.
- Extreme outliers with no explanatory context.

These series are labeled `low_confidence` in the recommendation system.

---

## Statistical summary (aggregated format)

### By cluster (see [06_clustering_recommendations.md](06_clustering_recommendations.md))

| Cluster | Best model (median MASE) | Median MASE |
|---|---|---|
| High volume | Chronos-Bolt | ~0.6–0.9 |
| Growth | Mixed (TimesFM ≈ TabPFN-TS) | ~0.8–1.1 |

### Wilcoxon corrected with Holm-Bonferroni (per family)

Out of the ~10 comparison pairs × 2 clusters = ~20 tests, typically **only 1 or 2 survive the Holm-Bonferroni correction** at `α = 0.05`. Most differences are **practically small** even if statistically marginal.

### Diebold-Mariano (month-by-month)

At the individual-series level, Chronos-Bolt wins in the majority within the "high volume" cluster (>70% of series with `p<0.05`). In the "growth" cluster the dominance is split.

---

## Limitations

1. **Tests with a moderate n** (tens of series per cluster). Small differences may not be detected statistically.
2. **Limited temporal coverage** (5 years): does not allow assessing behavior under multi-year shocks (licensing changes, open-access transition).
3. **No fine-tuning** of the foundation models. All results are zero-shot.
4. **No ensemble**: combining the predictions of several models was not evaluated exhaustively.

---

## Operational conclusion

For institutional use in a consortium:
- **Default recommended model**: Chronos-Bolt small (fast, competitive, no covariates).
- **Secondary model for validation**: TimesFM with institutional covariates (more interpretable when shocks occur).
- **PyCaret AutoML**: not recommended as a first line because of its computational cost without a consistent gain.
- **Final decision**: a mixed rule by cluster — see [06_clustering_recommendations.md](06_clustering_recommendations.md).
