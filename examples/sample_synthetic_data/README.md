# Synthetic-data generator

This directory contains instructions to generate a synthetic dataset with the same schema as the private consortium dataset, so the notebooks can be run end-to-end without access to real data.

## Generated schema

It exactly follows the schema in [`docs/01_dataset_overview.md`](../../docs/01_dataset_overview.md):

| Column | Type | Detail |
|---|---|---|
| `institution` | string | `Inst_01` … `Inst_NN` |
| `publisher` | string | `Pub_A` … `Pub_Z` |
| `report` | string | `TR_J1` |
| `date` | timestamp | Monthly, first day of the month |
| `metric_type` | string | `Total_Item_Requests` |
| `value` | int64 | Synthetic monthly downloads |

## Characteristics of the synthetic series

- **Annual seasonality** (troughs between June and August, peaks March–May and September–November).
- **Random trend** per series (positive, flat, or negative).
- **Volumes** log-normally distributed: approximate range 10–10⁵ downloads/month.
- **Gaussian noise** proportional to the volume.
- **Occasional outliers** (≤2% of months) to simulate promotions or contract changes.

## Generator

The runnable generator is [`generate_sample.py`](generate_sample.py). It also
produces the two covariate files required by notebooks 02 and 08:

| File | Schema |
|---|---|
| `anonymized_series.parquet` | `institution, publisher, report, date, metric_type, value` |
| `dynamic_covariates.parquet` | `institution, year, prop_area_1..K` (enrollment shares, sum 1 per row) |
| `static_covariates.parquet` | `publisher, kbart_area_1..K` (catalog shares, sum 1) + `dominant_sjr` |

## How to use

1. Run it from anywhere: `python examples/sample_synthetic_data/generate_sample.py`.
2. It writes the three parquet files directly into the repo's `data/` folder (root) — no manual copy or rename needed.
3. Run notebooks `01`–`08` in order, each in its corresponding virtual environment.

## Limitations

The synthetic series **do not replicate** the real dynamics (contract changes, institutional events, specific promotional spikes). Model results on synthetic data serve only as a pipeline functional test, not as evidence of performance.
