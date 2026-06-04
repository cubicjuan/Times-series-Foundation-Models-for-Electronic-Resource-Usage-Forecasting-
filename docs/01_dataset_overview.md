# 01 — Dataset schema (no data)

This document describes **the shape** of the dataset expected by the
notebooks. The real values are private and are not distributed.

## Source

- COUNTER 5 reports (the international standard for electronic-resource usage).
- Report types used: `TR_J1` (controlled journal usage) and `TR_J3` (usage by access type).
- Granularity: monthly.

## Logical schema

A single parquet file with the following columns:

| Column | Type | Description |
|---|---|---|
| `institution` | string | Anonymized identifier (`Inst_01` … `Inst_NN`) |
| `publisher` | string | Anonymized publishing-platform identifier (`Pub_A` … `Pub_Z`) |
| `report` | string | `TR_J1` or `TR_J3` |
| `date` | timestamp[ns] | First day of the month |
| `metric_type` | string | COUNTER metric (filtered to `Total_Item_Requests`) |
| `value` | int64 | Number of downloads in that month |

## Temporal coverage

- Range used for training: 5 years of monthly history (e.g. 2020-01 → 2024-12).
- Evaluation horizon: 12 out-of-sample months.
- Series per institution and publisher: lengths may differ. Minimum 24 months to enter comparisons; minimum 36 for PyCaret.

## Modeling-relevant characteristics

- **Dominant annual seasonality** (academic calendar): vacation troughs in Jul–Dec depending on the university calendar.
- **Highly uneven volume across series**: typical ranges between ~10 and ~10⁵ monthly downloads.
- **Intermittent series**: some `institution × publisher` combinations have months with a value of zero.
- **Occasional outliers** associated with promotional campaigns or contract changes.

## Standard filters

Before modeling, every notebook applies:

```python
df = df[df["report"].isin(["TR_J1", "TR_J3"])]
df = df[df["metric_type"].str.lower().str.contains("total.item", regex=True, na=False)]
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.year.between(2020, 2024)]
```

## Exogenous covariates (optional)

Some notebooks (TimesFM + cov, TabPFN-TS + cov) use external covariates that must also be anonymized or replaced.

### Dynamic (time-varying)

| Column | Frequency | Description |
|---|---|---|
| `prop_area_{1..N}` | annual | Share of institutional enrollment in each field of knowledge |

### Static (constant per series)

| Column | Description |
|---|---|
| `kbart_area_{1..N}` | Share of the publisher's catalog in each subject area (KBART) |
| `dominant_sjr` | Predominant SJR quartile (Q1–Q4) of the publisher |

The covariate schema is documented in each notebook that uses it.

## Synthetic data for reproducibility

`examples/sample_synthetic_data/` contains instructions to generate synthetic series with the same schema, annual seasonality, and plausible volume range. This makes it possible to run the notebooks end-to-end without access to real data.
