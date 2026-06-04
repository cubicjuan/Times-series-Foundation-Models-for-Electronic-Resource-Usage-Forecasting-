"""Synthetic-data generator for the forecasting pipeline.

Generates three parquet files with the same schema as the private consortium
dataset, so the notebooks can run end-to-end without access to real data:

  - anonymized_series.parquet   (institution, publisher, report, date, metric_type, value)
  - dynamic_covariates.parquet  (institution, year, prop_area_1..K)   [annual enrollment shares]
  - static_covariates.parquet   (publisher, kbart_area_1..K, dominant_sjr)

Usage:
    python generate_sample.py
    # then copy the three files into the repo's data/ folder (root).

The synthetic series do NOT replicate the real dynamics; they only exercise the
pipeline. Results on synthetic data are a functional test, not evidence of
performance.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Output: repo-root data/ folder (../../data relative to this file)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEED = 42
N_INSTITUTIONS = 10
N_PUBLISHERS = 5
N_AREAS = 4
YEAR_START = 2020
YEAR_END = 2024

rng = np.random.default_rng(SEED)
dates = pd.date_range(f"{YEAR_START}-01-01", f"{YEAR_END}-12-01", freq="MS")
n_months = len(dates)
years = list(range(YEAR_START, YEAR_END + 1))

# Base seasonal pattern (12 months, normalized)
base_seasonal = np.array([
    0.85, 0.90, 1.15, 1.20, 1.10, 0.75,
    0.65, 0.70, 1.25, 1.30, 1.15, 0.80,
])


def random_shares(n):
    """A vector of n positive shares that sums to 1."""
    x = rng.random(n) + 0.1
    return x / x.sum()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Usage series ----
    rows = []
    for i in range(N_INSTITUTIONS):
        inst = f"Inst_{i + 1:02d}"
        for j in range(N_PUBLISHERS):
            pub = f"Pub_{chr(ord('A') + j)}"
            base_volume = rng.lognormal(mean=5, sigma=1.5)
            trend = rng.normal(0, 0.005)
            noise_std = 0.15
            seasonal = np.tile(base_seasonal, n_months // 12 + 1)[:n_months]
            month_idx = np.arange(n_months)
            values = base_volume * seasonal * np.exp(trend * month_idx)
            values *= (1 + rng.normal(0, noise_std, n_months))
            outliers = rng.random(n_months) < 0.02
            values[outliers] *= rng.uniform(1.5, 3.0, outliers.sum())
            values = np.clip(values, 0, None).round().astype(int)
            for k, date in enumerate(dates):
                rows.append({
                    "institution": inst,
                    "publisher": pub,
                    "report": "TR_J1",
                    "date": date,
                    "metric_type": "Total_Item_Requests",
                    "value": int(values[k]),
                })
    series_df = pd.DataFrame(rows)
    series_df.to_parquet(DATA_DIR / "anonymized_series.parquet", index=False)

    # ---- 2. Dynamic covariates: enrollment shares per (institution, year) ----
    dyn_rows = []
    for i in range(N_INSTITUTIONS):
        inst = f"Inst_{i + 1:02d}"
        shares = random_shares(N_AREAS)
        for year in years:
            # small year-to-year drift, renormalized
            drift = shares * (1 + rng.normal(0, 0.05, N_AREAS))
            drift = np.clip(drift, 1e-6, None)
            drift = drift / drift.sum()
            row = {"institution": inst, "year": year}
            row.update({f"prop_area_{a + 1}": float(drift[a]) for a in range(N_AREAS)})
            dyn_rows.append(row)
    pd.DataFrame(dyn_rows).to_parquet(DATA_DIR / "dynamic_covariates.parquet", index=False)

    # ---- 3. Static covariates: catalog shares + dominant SJR per publisher ----
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    stat_rows = []
    for j in range(N_PUBLISHERS):
        pub = f"Pub_{chr(ord('A') + j)}"
        shares = random_shares(N_AREAS)
        row = {"publisher": pub}
        row.update({f"kbart_area_{a + 1}": float(shares[a]) for a in range(N_AREAS)})
        row["dominant_sjr"] = str(rng.choice(quartiles))
        stat_rows.append(row)
    pd.DataFrame(stat_rows).to_parquet(DATA_DIR / "static_covariates.parquet", index=False)

    print(f"Generated in {DATA_DIR}:")
    print(f"  anonymized_series.parquet   ({len(series_df)} rows)")
    print(f"  dynamic_covariates.parquet  ({len(dyn_rows)} rows)")
    print(f"  static_covariates.parquet   ({len(stat_rows)} rows)")


if __name__ == "__main__":
    main()
