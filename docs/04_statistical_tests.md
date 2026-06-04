# 04 — Statistical tests

Comparing two models by their median MASE is not enough: one must show that the difference is **statistically significant** and not a product of the between-series variance. Three complementary tests are used.

---

## 1. Wilcoxon signed-rank (on per-series MASE)

**Question:** does model A produce a systematically lower MASE than model B over the set of series?

**Test form:** non-parametric, paired, on the difference `MASE_A(series) - MASE_B(series)` for each series.

**Implementation:**

```python
from scipy.stats import wilcoxon

stat, p = wilcoxon(mase_model_a, mase_model_b, alternative="less")
```

`alternative="less"` for a directed hypothesis: model A is better (lower MASE) than B.

**Report:**
- `n` = number of series with both models available
- `wins_A` = number of series where A < B
- `wins_B` = number of series where A > B
- `effect_size_r` = `|Z| / sqrt(n)`
- `p_value`

**Effect-size interpretation:**
- `r < 0.1` → trivial
- `0.1 ≤ r < 0.3` → small
- `0.3 ≤ r < 0.5` → medium
- `r ≥ 0.5` → large

---

## 2. Diebold-Mariano (month-by-month)

**Question:** does model A have significantly smaller errors than model B over the full horizon, accounting for autocorrelation?

**Test form:** on the difference of monthly losses `d_t = L(e_A,t) - L(e_B,t)` with `L(e) = e²` (squared loss). It corrects for autocorrelation in the loss residual.

**Implementation:**

```python
def diebold_mariano(e_a, e_b, h=1, loss="mse"):
    if loss == "mse":
        d = e_a**2 - e_b**2
    elif loss == "mae":
        d = np.abs(e_a) - np.abs(e_b)
    n = len(d)
    mean_d = d.mean()
    # Variance with Newey-West correction up to lag h-1
    gamma_0 = np.var(d, ddof=0)
    var = gamma_0
    for k in range(1, h):
        gamma_k = ((d[k:] - mean_d) * (d[:-k] - mean_d)).mean()
        var += 2 * (1 - k / h) * gamma_k
    dm = mean_d / np.sqrt(var / n)
    p = 2 * (1 - scipy.stats.norm.cdf(abs(dm)))
    return dm, p
```

**Per series:** applied to the `h = 12` monthly test errors. Per family: aggregated by counting how many series reject H₀ in favor of model A.

---

## 3. Holm-Bonferroni correction (per family of tests)

**Problem:** comparing `k` models pairwise involves `k(k-1)/2` tests. Without correction, the false-positive risk inflates.

**For 5 compared models:**
- Pairs: `C(5, 2) = 10` tests
- Separate families: one per cluster, one per publisher

**Implementation:**

```python
from statsmodels.stats.multitest import multipletests

reject, p_adj, _, _ = multipletests(
    pvals=p_values,
    alpha=0.05,
    method="holm",
)
```

Holm is more powerful than the classical Bonferroni while still controlling the FWER (Family-Wise Error Rate).

---

## Reporting rules

1. **Every model comparison** reports Wilcoxon and DM side by side.
2. **The test does not replace the metric.** The median MASE + IQR is reported alongside the p-value.
3. **Significant differences with a small effect size** are mentioned but not over-interpreted.
4. **Pairs with `n < 10` common series** are excluded from the test (insufficient statistical power).
5. **Series with extreme MASE values** (typically `MASE > 5` due to data-quality issues) are reported separately and documented.

## Reference implementation

See [src/statistical_tests.py](../src/statistical_tests.py). The notebooks must import from there.
