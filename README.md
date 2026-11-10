# Retail-360: Customer Lifecycle & Neural LTV Engine

[![Retail-360 CI Pipeline](https://github.com/mickeywmoore/Retail-360-Customer-Lifecycle-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/mickeywmoore/Retail-360-Customer-Lifecycle-Engine/actions/workflows/ci.yml)

An end-to-end Python CLI and data engineering pipeline that transforms relational e-commerce transaction logs into an RFM (Recency, Frequency, Monetary) feature store, segments customer personas via K-Means clustering, forecasts 90-day Customer Lifetime Value (LTV) with a multi-layer neural regressor, and audits regional error disparity for algorithmic fairness.

---

## Architecture & Pipeline Stages

1. **Relational SQL Feature Store (`src/retail360/data_store.py`)**
   * Seeds or ingests Olist-compatible relational tables (`orders`, `customers`, `payments`, `reviews`) in SQLite.
   * Executes multi-CTE SQL aggregations to resolve unique customer identities, filter non-delivered orders, and compute customer-level `recency_days`, `frequency`, `monetary_value`, and review scores.
2. **Unsupervised Persona Segmentation (`src/retail360/segmentation.py`)**
   * Applies `log1p` skewness correction and `StandardScaler` normalization before fitting `k-means++` clustering and computing WCSS elbow diagnostics.
3. **90-Day Neural LTV Forecasting (`src/retail360/ltv_model.py`)**
   * Trains a `64 -> 32 -> 16` neural regression architecture (supporting both Scikit-learn `MLPRegressor` with early stopping and optional TensorFlow/Keras backends) with strict train/test scaling isolation to prevent data leakage.
4. **Regional Fairness & Bias Auditor (`src/retail360/fairness.py`)**
   * Evaluates subgroup Mean Absolute Error (MAE) across geographic cohorts (`North`, `South`, `East`, `West`) and enforces configurable maximum disparity thresholds to prevent regional redlining in retention spend.

---

## Repository Layout

```text
Retail-360-Customer-Lifecycle-Engine/
├── pyproject.toml
├── README.md
├── src/
│   └── retail360/
│       ├── __init__.py
│       ├── cli.py          # Terminal CLI entry point & JSON report exporter
│       ├── data_store.py   # SQLite schema loader & multi-CTE RFM feature query
│       ├── fairness.py     # Regional subgroup MAE disparity auditor
│       ├── ltv_model.py    # Neural network LTV regressor (MLP / Keras)
│       └── segmentation.py # Log-scaled K-Means clustering & WCSS diagnostics
└── tests/
    └── test_pipeline.py    # Automated unit & end-to-end CLI integration tests
```

---

## Quickstart (CLI & Tests)

```bash
# Create virtual environment and install package in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the full pipeline in the terminal and export structured JSON metrics
retail360 run-pipeline --orders 10000 --customers 2000 --clusters 4 --output-json artifacts/report.json

# Run the automated test suite
python -m unittest discover -s tests -v
```

---

## Active Development Roadmap (v2.x)

- [ ] Add CSV/Parquet ingestion subcommand (`retail360 ingest`) with strict schema validation for external transaction exports.
- [ ] Implement model artifact persistence (`joblib` / Keras `.keras`) and standalone `retail360 predict` batch inference CLI command.
- [ ] Add Population Stability Index (PSI) feature drift detection between baseline training distributions and new scoring batches.
- [ ] Expand fairness auditing across multi-dimensional cohorts with automated Markdown/HTML compliance report generation.
