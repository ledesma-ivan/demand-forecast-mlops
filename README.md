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
| Orchestration | Docker Compose | Training job + MLflow UI |
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
├── Dockerfile
├── docker-compose.yml         # Orchestration: training job + MLflow UI
├── docker-compose.spark.yml   # Orchestration: PySpark feature engineering cluster
└── pyproject.toml             # Ruff and tooling config
```

---

## Environment Variables

Create a `.env` file in the project root before running:

```
GOOGLE_API_KEY=your_api_key_here
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

`GOOGLE_API_KEY` is required for the `/report` endpoint (Gemini AI). The rest of the pipeline runs without it.

---

## Quick Start

### Option A — Docker (recommended)

#### MLflow training pipeline

```bash
git clone https://github.com/ledesma-ivan/demand-forecast-mlops.git
cd demand-forecast-mlops

dvc pull

docker compose up --build
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |

To run only the training job:

```bash
docker compose run training_job
```

#### dbt + DuckDB transformation pipeline

```bash
pip install -r requirements/dbt.txt

python dbt/load_sources.py

dbt run --profiles-dir dbt --project-dir dbt

dbt test --profiles-dir dbt --project-dir dbt

python dbt/export_mart.py

# Generate lineage graph
dbt docs generate --profiles-dir dbt --project-dir dbt
dbt docs serve --profiles-dir dbt --project-dir dbt
```

| Model | Layer | Description |
|-------|-------|-------------|
| `stg_sales` | staging | Cleaned weekly sales (types, renamed columns) |
| `stg_features` | staging | Store features with markdown nulls → 0 |
| `stg_stores` | staging | Store metadata + ordinal type encoding |
| `int_features` | intermediate | Lags (1/2/4/8/52w), rolling mean/std/max (4/8/12w), cross-series rank |
| `mart_training_set` | marts | Final table filtered to rows with full lag history; exported as Parquet |

---

#### PySpark feature engineering pipeline

```bash
docker-compose -f docker-compose.spark.yml up --abort-on-container-exit
```

| Service | URL |
|---------|-----|
| Spark Master UI | http://localhost:8080 |

Reads `data/train.csv`, `data/features.csv` and `data/stores.csv`, computes all features using the Spark DataFrame API, and writes the result to `data/processed/spark_features/` as Parquet partitioned by `Store` and `Week`. The cluster shuts down automatically when the job finishes.

---

### Option B — Local

```bash
git clone https://github.com/ledesma-ivan/demand-forecast-mlops.git
cd demand-forecast-mlops

python3.11 -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

pip install -r requirements.txt

dvc pull

python run_pipeline.py

uvicorn src.api.main:app --reload

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
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
- **Models evaluated:** XGBoost as baseline, Prophet for seasonality and trend, LSTM with TensorFlow for non-linear sequential patterns
- **Evaluation metrics:** RMSE and MAPE per model, automatically logged in MLflow for cross-run comparison
- **Storage:** SQLite for MLflow experiment persistence and data querying via SQL
- **Visualization:** Matplotlib for EDA in notebooks, Plotly for interactive model comparison vs real sales
- **LLM reporting:** Gemini AI (`gemini-3.1-flash-lite-preview`) to automatically generate natural language reports from MLflow run metrics

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
