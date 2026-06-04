"""Standardized loading of monthly COUNTER 5 usage series.

This module defines the single loading interface used by every notebook.
It expects a parquet file following the schema documented in
docs/01_dataset_overview.md.

Real institution and publisher identifiers never appear in code: the
dataset must be delivered already carrying anonymized labels
(Inst_01, Pub_A, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "institution",
    "publisher",
    "report",
    "date",
    "metric_type",
    "value",
]

VALID_REPORTS = ("TR_J1", "TR_J3")
METRIC_TOTAL_ITEM = "total.item"
SEASONAL_PERIOD = 12


@dataclass
class MonthlySeries:
    """A single monthly time series ready to feed a model."""

    series_id: str            # e.g. "Inst_01__Pub_A"
    institution: str
    publisher: str
    dates: pd.DatetimeIndex
    values: np.ndarray        # float32, length = len(dates)

    @property
    def n_months(self) -> int:
        return len(self.values)

    def split_train_test(self, horizon: int = 12) -> tuple[np.ndarray, np.ndarray]:
        if self.n_months <= horizon:
            raise ValueError(
                f"Series {self.series_id} has {self.n_months} months, "
                f"insufficient for horizon={horizon}"
            )
        train = self.values[:-horizon]
        test = self.values[-horizon:]
        return train, test


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load the parquet file applying the project's standard filters.

    Applies:
      - reads only the required columns
      - filters by report (TR_J1, TR_J3)
      - filters by metric_type (Total Item Requests)
      - converts the date column to datetime
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Does not exist: {path}")

    df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    df = df[df["report"].isin(VALID_REPORTS)]
    mask_metric = (
        df["metric_type"].str.lower().str.contains(METRIC_TOTAL_ITEM, regex=True, na=False)
    )
    df = df[mask_metric].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["institution", "publisher", "date"])
    return df.reset_index(drop=True)


def filter_period(df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Filter rows between year_start and year_end (both inclusive)."""
    mask = df["date"].dt.year.between(year_start, year_end)
    return df.loc[mask].copy()


def build_series(
    df: pd.DataFrame,
    min_months: int = 24,
) -> dict[str, MonthlySeries]:
    """Build a dict of MonthlySeries keyed by (institution, publisher).

    Sums values per month when several reports exist for the same series
    (TR_J1 + TR_J3) and drops series with fewer than `min_months` points.
    """
    agg = (
        df.groupby(["institution", "publisher", "date"], as_index=False)["value"]
        .sum()
    )

    series: dict[str, MonthlySeries] = {}
    for (inst, pub), group in agg.groupby(["institution", "publisher"]):
        group = group.sort_values("date")
        if len(group) < min_months:
            continue
        sid = f"{inst}__{pub}"
        series[sid] = MonthlySeries(
            series_id=sid,
            institution=inst,
            publisher=pub,
            dates=pd.DatetimeIndex(group["date"].values),
            values=group["value"].astype(np.float32).values,
        )
    return series


def input_matrix(
    series: dict[str, MonthlySeries],
    horizon: int = 12,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Return parallel lists (train, test, series_id) for batch inference."""
    train_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []
    ids: list[str] = []
    for sid, series_obj in series.items():
        try:
            train, test = series_obj.split_train_test(horizon=horizon)
        except ValueError:
            continue
        train_list.append(train.astype(np.float32))
        test_list.append(test.astype(np.float32))
        ids.append(sid)
    return train_list, test_list, ids
