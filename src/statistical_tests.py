"""Statistical tests for model comparison.

Includes:
  - Wilcoxon signed-rank test (on per-series MASE)
  - Diebold-Mariano test (on month-by-month errors)
  - Holm-Bonferroni correction (per family of tests)

See docs/04_statistical_tests.md for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class WilcoxonResult:
    n: int
    wins_a: int
    wins_b: int
    ties: int
    statistic: float
    p_value: float
    effect_size_r: float


def wilcoxon_paired(
    mase_a: np.ndarray,
    mase_b: np.ndarray,
    alternative: str = "less",
) -> WilcoxonResult:
    """Paired Wilcoxon test on the MASE of two models.

    alternative='less' tests that model A has a lower MASE than model B.
    """
    a = np.asarray(mase_a, dtype=np.float64)
    b = np.asarray(mase_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Arrays of different size")

    diff = a - b
    wins_a = int(np.sum(diff < 0))
    wins_b = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    n = len(diff)

    stat, p = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    z = stats.norm.ppf(1 - p) if alternative == "less" else stats.norm.ppf(p)
    effect_r = abs(z) / np.sqrt(n) if n > 0 else float("nan")

    return WilcoxonResult(
        n=n,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        statistic=float(stat),
        p_value=float(p),
        effect_size_r=float(effect_r),
    )


@dataclass
class DMResult:
    dm_stat: float
    p_value: float
    loss: str
    h: int


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "mse",
) -> DMResult:
    """Diebold-Mariano test for two aligned error series.

    Args:
        errors_a, errors_b: residuals (y_true - y_pred) per month.
        h: forecast step (horizon used for the Newey-West correction).
        loss: 'mse' or 'mae'.
    """
    e_a = np.asarray(errors_a, dtype=np.float64)
    e_b = np.asarray(errors_b, dtype=np.float64)
    if e_a.shape != e_b.shape:
        raise ValueError("Errors of different size")

    if loss == "mse":
        d = e_a ** 2 - e_b ** 2
    elif loss == "mae":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"Unsupported loss: {loss}")

    n = len(d)
    mean_d = d.mean()

    gamma_0 = np.var(d, ddof=0)
    var = gamma_0
    for k in range(1, h):
        gamma_k = ((d[k:] - mean_d) * (d[:-k] - mean_d)).mean()
        var += 2 * (1 - k / h) * gamma_k

    se = np.sqrt(var / n)
    if se < 1e-12:
        return DMResult(dm_stat=0.0, p_value=1.0, loss=loss, h=h)

    dm = mean_d / se
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return DMResult(dm_stat=float(dm), p_value=float(p), loss=loss, h=h)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni correction. Returns rejection flags and adjusted p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    p_ord = p[order]
    p_adjusted_ord = np.minimum.accumulate((n - np.arange(n)) * p_ord)[::-1][::-1]
    p_adjusted_ord = np.minimum(p_adjusted_ord, 1.0)
    p_adjusted = np.empty_like(p_adjusted_ord)
    p_adjusted[order] = p_adjusted_ord
    reject = p_adjusted < alpha
    return {
        "p_adjusted": p_adjusted.tolist(),
        "reject": reject.tolist(),
        "alpha": alpha,
    }
