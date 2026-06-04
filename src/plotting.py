"""Standard project figures.

Every function saves a PNG to disk (it does not display inline) to keep
the notebooks lightweight and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 150
FIGSIZE_SERIES = (10, 4)
FIGSIZE_HEATMAP = (10, 6)
FIGSIZE_BOX = (8, 5)


def plot_series_with_forecast(
    train_dates: pd.DatetimeIndex,
    train_values: np.ndarray,
    test_dates: pd.DatetimeIndex,
    test_values: np.ndarray,
    forecasts: dict[str, np.ndarray],
    title: str,
    output_path: str | Path,
) -> None:
    """Plot train + actual test + forecasts of one or several models."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SERIES)
    ax.plot(train_dates, train_values, color="gray", label="History (train)")
    ax.plot(test_dates, test_values, color="black", linewidth=2, label="Actual (test)")
    for name, pred in forecasts.items():
        ax.plot(test_dates, pred, linestyle="--", label=name)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly downloads")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_metric_boxplot(
    metrics_df: pd.DataFrame,
    metric: str,
    model_column: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Boxplot of a metric by model (one box per model)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_BOX)
    models = sorted(metrics_df[model_column].unique())
    data = [metrics_df.loc[metrics_df[model_column] == m, metric].values for m in models]
    ax.boxplot(data, labels=models, showfliers=False)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="Seasonal Naive (MASE=1)")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: str | Path,
    cmap: str = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Heatmap (rows x columns). Numeric values centered in each cell."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_scatter_clusters(
    pca_components: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """2-D scatter colored by cluster (projected onto 2 components)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_clusters = sorted(np.unique(labels))
    for c in unique_clusters:
        mask = labels == c
        ax.scatter(
            pca_components[mask, 0],
            pca_components[mask, 1],
            label=f"Cluster {c}",
            alpha=0.7,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
