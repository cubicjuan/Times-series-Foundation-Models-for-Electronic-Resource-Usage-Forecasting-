# Times-series-Foundation-Models-for-Electronic-Resource-Usage-Forecasting-
Forecasting monthly usage of academic databases (COUNTER 5) with time-series foundation models — TimesFM, Chronos-Bolt and TabPFN-TS — benchmarked against PyCaret AutoML and a Seasonal Naive baseline.
## Objective

Evaluate, compare, and combine modern univariate forecasting models on monthly download series (COUNTER 5 metric — Total Item Requests) to support renewal, renegotiation, and subscription-deepening decisions for electronic resources in a consortium of university libraries.

## Models compared

| Model | Type | Exogenous cov. | Source |
|---|---|---|---|
| **TimesFM 2.5 (200M)** | Foundation (Google) | Yes (`forecast_with_covariates`) | `google/timesfm-2.5-200m-pytorch` |
| **Chronos-Bolt** | Foundation (Amazon) | No | `amazon/chronos-bolt-small` |
| **TabPFN-TS** | Foundation (Prior Labs) | Yes (optional) | `tabpfn-time-series` |
| **PyCaret AutoML** | Classical (best per series) | No | `pycaret[time-series]` |
| **Seasonal Naive** | Baseline | No | in-house implementation |

## Repository structure

```
.
├── README.md                       This file
├── LICENSE                         CC-BY-4.0
├── requirements.txt                Main environment (TimesFM, PyCaret)
├── requirements-chronos.txt        Isolated environment for Chronos
├── requirements-tabpfn.txt         Isolated environment for TabPFN-TS
│
├── docs/
│   ├── 01_dataset_overview.md      Data schema (anonymized)
│   ├── 02_models.md                Architecture and configuration of each model
│   ├── 03_evaluation_protocol.md   Train/test split, metrics
│   ├── 04_statistical_tests.md     Diebold-Mariano, Wilcoxon, Holm-Bonferroni
│   ├── 05_results.md               Aggregated results (no identifiers)
│   └── 06_clustering_recommendations.md
│
├── notebooks/
│   ├── 01_pipeline_timesfm.ipynb              TimesFM (univariate)
│   ├── 02_pipeline_timesfm_covariates.ipynb   TimesFM + covariates
│   ├── 03_chronos_inference.ipynb             Chronos-Bolt
│   ├── 04_tabpfn_ts.ipynb                     TabPFN-TS
│   ├── 05_pycaret_baseline.ipynb              PyCaret AutoML + Seasonal Naive
│   ├── 06_multimodel_comparison.ipynb         Unified metrics + statistical tests
│   ├── 07_clustering_series.ipynb             K-Means usage profiles
│   └── 08_recommendation_system.ipynb         Decision rule per series
│
├── src/
│   ├── data_loader.py              Standardized loading of monthly series
│   ├── metrics.py                  MASE, MAE, RMSE, sMAPE, MedAE, MAPE
│   ├── statistical_tests.py        DM, Wilcoxon, Holm-Bonferroni
│   └── plotting.py                 Standard figures
│
├── examples/
│   └── sample_synthetic_data/      Synthetic-data generator (reproducibility)
│       └── generate_sample.py      Produces series + covariates parquet files
└── CITATION.cff                    Citation metadata
```

The notebooks form a sequence: **01–05** generate per-model forecasts, **06**
unifies and tests them statistically, **07** clusters the series, and **08**
turns everything into a per-series subscription recommendation.

## Note on data

**The notebooks do NOT include real consortium data.** Institution identifiers appear as `Inst_01`, `Inst_02`, … and publishers as `Pub_A`, `Pub_B`, … The expected data structure is documented in [docs/01_dataset_overview.md](docs/01_dataset_overview.md) and can be reproduced with the synthetic generator in `examples/sample_synthetic_data/`.

## Installation

Three separate virtual environments are recommended (their dependencies conflict):

```bash
# Main environment — TimesFM + PyCaret
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Chronos environment
python -m venv .venv_chronos
.venv_chronos\Scripts\activate
pip install -r requirements-chronos.txt

# TabPFN environment
python -m venv .venv_tabpfn
.venv_tabpfn\Scripts\activate
pip install -r requirements-tabpfn.txt
```

Python 3.10.x. PyTorch with CUDA recommended for Chronos and TabPFN-TS.

## How to run the notebooks

Each notebook is **an output-free template** documenting the workflow:
1. Dataset loading (documented schema, no real values)
2. Model configuration
3. Inference
4. Computation of the 6 standard metrics
5. Comparison against the baseline

To reproduce with your own data:
1. Place your parquet file in `data/` following the schema in [docs/01_dataset_overview.md](docs/01_dataset_overview.md)
2. Adjust the paths in the configuration cell of each notebook
3. Run it in the corresponding virtual environment

To run end-to-end with synthetic data (no real data required):
1. `python examples/sample_synthetic_data/generate_sample.py` — writes
   `anonymized_series.parquet`, `dynamic_covariates.parquet` and
   `static_covariates.parquet` into `data/`.
2. Run notebooks `01`–`08` in order (notebook 05 needs the main `.venv`;
   notebooks 03 and 04 use their isolated environments).

## Standard metrics

Each evaluation reports the **six metrics** documented in [docs/03_evaluation_protocol.md](docs/03_evaluation_protocol.md):

- **MASE** — primary metric (scaled relative to the seasonal naive)
- **MAE** — absolute error in real units
- **RMSE** — penalizes large errors
- **sMAPE** — symmetric percentage error, robust to zeros
- **MedAE** — median of the absolute error
- **MAPE** — only when `y_true > 0`

## Statistical tests

To compare models pairwise the following are used:

- **Diebold-Mariano** at the month-by-month level
- **Wilcoxon signed-rank** on the per-series MASE
- **Holm-Bonferroni correction** per family of tests

Details in [docs/04_statistical_tests.md](docs/04_statistical_tests.md).

## Citation

If you use this work in publications, please cite the corresponding master's thesis.

## License

CC-BY-4.0 — see [LICENSE](LICENSE).
