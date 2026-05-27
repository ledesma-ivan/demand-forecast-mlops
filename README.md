# Demand Forecast MLOps

End-to-end ML pipeline for retail demand forecasting — from raw data to a production-ready REST API, with full experiment tracking via MLflow.

---

## Business Impact & Metrics

| Metric | Result |
|--------|--------|
| LSTM RMSE | 1,098.77 |
| XGBoost MAPE | 4.0% |
| Forecast horizon | Weekly (52-week lag features) |
| Business goal | Reduce stockouts and excess inventory via accurate demand prediction |

> Accurate demand forecasting directly impacts inventory decisions: lower MAPE reduces overstock risk, while lower RMSE minimizes costly stockout events across store-department combinations.

---

## MLOps Architecture

```
Raw Data (DVC)
     │
     ├──► Feature Pipeline (Pandas)  ──► Feature Store (Parquet)
     │
     ├──► PySpark Feature Pipeline   ──► Partitioned Parquet (Store/Week)
     │
     └──► dbt + DuckDB Pipeline      ──► mart_training_set.parquet
               stg_sales.sql                    │
               stg_features.sql                 │
               stg_stores.sql                   │
               int_features.sql   (window fns)  │
               mart_training_set.sql             │
                                                 │
     ┌───────────────────────────────────────────┘
     ▼
Model Training ──► MLflow Experiment Tracking
  ├── XGBoost
  ├── Prophet
  └── LSTM (TensorFlow)
     │
     ▼
Model Registry (MLflow)
     │
     ▼
REST API (FastAPI + Docker)
     │
     ▼
LLM Reporting (Gemini AI → /report endpoint)
```

| Layer | Tool | Role |
|-------|------|------|
| Data versioning | DVC | Raw & processed data tracking |
| Feature engineering (scale) | PySpark + Docker | Window functions, lag & rolling features at scale |
| Feature engineering (SQL) | dbt Core + DuckDB | Versioned, tested SQL transformation layer |
| Feature store | Parquet (PyArrow) | Feature persistence and reuse across runs |
| Experiment tracking | MLflow + SQLite | Parameters, metrics and artifact logging |
| Model registry | MLflow | Versioned model storage and promotion |
| Serving | FastAPI + Uvicorn | REST predictions endpoint |
| Orchestration | Docker Compose | Full pipeline: dbt, PySpark, MLflow and training in one command |
| LLM reporting | Gemini AI | Natural language summaries from run metrics |

---

## Tech Stack

`Python` · `Pandas` · `PySpark` · `dbt` · `DuckDB` · `NumPy` · `PyArrow` · `SQLite` · `Scikit-learn` · `XGBoost` · `Prophet` · `TensorFlow` · `Matplotlib` · `Plotly` · `DVC` · `MLflow` · `Gemini AI` · `FastAPI` · `Uvicorn` · `Docker` · `Pytest` · `Ruff` · `pre-commit` · `GitHub Actions`

---

## Project Structure

```
demand-forecast-mlops/
├── .github/workflows/         # CI/CD pipelines
├── data/                      # Raw and processed datasets (managed by DVC)
├── notebooks/                 # EDA and experimentation
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── load_sources.py        # Loads CSVs into DuckDB before dbt run
│   ├── export_mart.py         # Exports mart to Parquet for MLflow
│   └── models/
│       ├── staging/           # stg_sales, stg_features, stg_stores
│       ├── intermediate/      # int_features (lags, rolling, cross-series)
│       └── marts/             # mart_training_set
├── spark/
│   └── feature_pipeline.py   # PySpark feature engineering job
├── src/
│   ├── api/                   # FastAPI endpoints
│   ├── features/              # Feature store and preprocessing (Pandas)
│   └── models/                # Training scripts and MLflow integration
├── tests/                     # pytest test suite
├── models/                    # Serialized trained models
├── mlruns/                    # MLflow artifacts
├── requirements/              # Environment-specific dependencies
├── run_pipeline.py            # Entrypoint: train, evaluate and register
├── dvc.yaml                   # DVC pipeline definition
├── Dockerfile                 # Training + MLflow image
├── Dockerfile.dbt             # Lightweight dbt + DuckDB image
├── docker-compose.yml         # Full pipeline: dbt, PySpark, MLflow, training
└── pyproject.toml             # Ruff and tooling config
```

---

## Environment Variables

Create a `.env` file in the project root before running:

```
GEMINI_API_KEY=your_api_key_here
```

`GEMINI_API_KEY` is required for the `/report` endpoint (Gemini AI). The rest of the pipeline runs without it.

---

## Quick Start

### Option A — Docker (recommended)

One command runs the full pipeline: dbt + DuckDB, PySpark and model training, all orchestrated automatically.

```bash
git clone https://github.com/ledesma-ivan/demand-forecast-mlops.git
cd demand-forecast-mlops

# Pull data (requires DVC configured) — or place CSVs manually in data/
dvc pull

docker compose up --build
```

**Execution order (managed by Docker Compose healthchecks):**

```
dbt_job ──────────────────────────┐
                                  ▼
spark-master → spark-worker → spark-job ──► training_job
                                  ▲
mlflow_ui (healthcheck OK) ───────┘
```

`dbt_job` and `spark-job` run in parallel. `training_job` starts only after both complete successfully and MLflow is ready.

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Spark Master UI | http://localhost:8080 |

**Outputs after a full run:**

| Path | Producer | Content |
|------|----------|---------|
| `data/processed/dbt_features/mart_training_set.parquet` | dbt + DuckDB | SQL-transformed feature set |
| `data/processed/spark_features/` | PySpark | Parquet partitioned by Store/Week |
| `mlruns/` | MLflow | Experiment runs, metrics, model artifacts |

To run only the training job (skipping dbt and Spark):

```bash
docker compose run training_job
```

---

### dbt lineage graph (optional)

```bash
pip install -r requirements/dbt.txt

dbt docs generate --profiles-dir dbt --project-dir dbt
dbt docs serve   --profiles-dir dbt --project-dir dbt
```

| Model | Layer | Description |
|-------|-------|-------------|
| `stg_sales` | staging | Cleaned weekly sales (types, renamed columns) |
| `stg_features` | staging | Store features with markdown nulls → 0 |
| `stg_stores` | staging | Store metadata + ordinal type encoding |
| `int_features` | intermediate | Lags (1/2/4/8/52w), rolling mean/std/max (4/8/12w), cross-series rank |
| `mart_training_set` | marts | Final table filtered to rows with full lag history; exported as Parquet |

---

### Option B — Local

```bash
git clone https://github.com/ledesma-ivan/demand-forecast-mlops.git
cd demand-forecast-mlops

python3.11 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt

dvc pull

# Terminal 1 — MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Terminal 2 — training pipeline
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python run_pipeline.py

# Terminal 3 — REST API
uvicorn src.api.main:app --reload
```

| Service | URL |
|---------|-----|
| API | http://127.0.0.1:8000 |
| Swagger docs | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

---

## API Usage

### `POST /predict` — demand forecast for a store/department/date

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"store": 1, "dept": 1, "date": "2010-11-26"}'
```

```json
{
  "store": 1,
  "dept": 1,
  "date": "2010-11-26",
  "predicted_demand": 24503.87
}
```

### `GET /report` — natural language summary of the latest MLflow run via Gemini AI

```bash
curl http://127.0.0.1:8000/report
```

Explore all endpoints interactively at **http://127.0.0.1:8000/docs** (FastAPI Swagger UI).

---

## Model Results

| Model   | RMSE      | MAPE | Training Time |
|---------|-----------|------|---------------|
| LSTM    | 1,098.77  | 5.1% | ~1m 10s       |
| XGBoost | 1,640.57  | 4.0% | ~11s          |
| Prophet | 2,313.03  | 9.4% | ~19s          |

LSTM achieves the lowest RMSE while XGBoost leads on MAPE and training efficiency (~8x faster). For production use cases requiring frequent retraining, XGBoost offers the best accuracy/speed trade-off. All runs are tracked and compared via MLflow.

---

## Technical Decisions

- **EDA:** Exploratory analysis on the Walmart dataset to identify weekly seasonality, holiday impact and sales variability across departments
- **Feature store:** Local implementation with Parquet (PyArrow) for feature persistence and reuse between runs
- **Feature engineering (Pandas):** Temporal features (week, month, quarter, year-end), markdowns, sales lags (1, 2, 4, 8, 52 weeks), rolling statistics (mean, std, max), and cross-store/department features
- **Feature engineering (PySpark):** Full re-implementation using the Spark DataFrame API — window functions for lag features (`lag(n).over`), rolling aggregations (`rowsBetween(-w, -1)`), and cross-series features (`dense_rank().over`); output as Parquet partitioned by Store/Week
- **Feature engineering (dbt + DuckDB):** SQL-native transformation layer with dbt Core and DuckDB adapter — three model layers (staging/intermediate/marts), declarative schema tests (`not_null`, `unique`, `accepted_values`), and `dbt docs` lineage graph; output as Parquet consumed by the existing MLflow pipeline
- **Orchestration:** Single `docker compose up --build` runs dbt, PySpark and model training with dependency ordering enforced by Docker Compose healthchecks and `service_completed_successfully` conditions
- **MLflow readiness:** `training_job` polls the MLflow `/health` endpoint (up to 60s) instead of a fixed sleep, so startup adapts to the actual server boot time
- **Models evaluated:** XGBoost as baseline, Prophet for seasonality and trend, LSTM with TensorFlow for non-linear sequential patterns
- **Evaluation metrics:** RMSE and MAPE per model, automatically logged in MLflow for cross-run comparison
- **Storage:** SQLite for MLflow experiment persistence and data querying via SQL
- **Visualization:** Matplotlib for EDA in notebooks, Plotly for interactive model comparison vs real sales
- **LLM reporting:** Gemini AI to automatically generate natural language reports from MLflow run metrics

---

## Code Quality & Testing

| Tool | Purpose |
|------|---------|
| `pytest` | Unit and integration testing |
| `pytest-cov` | Test coverage reporting |
| `Ruff` | Fast linting and formatting (replaces flake8 + isort) |
| `pre-commit` | Automatic checks before every commit |
| `GitHub Actions` | CI runs on every push |

```bash
pytest

pytest --cov=src tests/

ruff check --fix .
ruff format .
```

---

## CI/CD Pipeline

Automated pipeline via **GitHub Actions** triggered on every push to `main`:

| Stage | Tool | Description |
|-------|------|-------------|
| Code quality | Ruff + pre-commit | PEP8 style and import ordering |
| Testing | pytest + pytest-cov | Full test suite with coverage report |
| Build | Docker | Container image built and validated |
