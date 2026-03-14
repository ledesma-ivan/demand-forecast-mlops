# 🛒 Demand Forecast MLOps

End-to-end ML pipeline for retail demand forecasting — from raw data to a production-ready REST API, with full experiment tracking via MLflow.

---

## 📊 Business Impact & Metrics

| Metric | Result |
|--------|--------|
| LSTM RMSE | 1,098.77 |
| XGBoost MAPE | 4.0% |
| Forecast horizon | Weekly (52-week lag features) |
| Business goal | Reduce stockouts and excess inventory via accurate demand prediction |

> Accurate demand forecasting directly impacts inventory decisions: lower MAPE reduces overstock risk, while lower RMSE minimizes costly stockout events across store-department combinations.

---

## 🏗️ MLOps Architecture

```
Raw Data (DVC)
     │
     ▼
Feature Pipeline ──► Feature Store (Parquet)
     │
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
| Feature store | Parquet (PyArrow) | Feature persistence and reuse across runs |
| Experiment tracking | MLflow + SQLite | Parameters, metrics and artifact logging |
| Model registry | MLflow | Versioned model storage and promotion |
| Serving | FastAPI + Uvicorn | REST predictions endpoint |
| Orchestration | Docker Compose | Training job + MLflow UI |
| LLM reporting | Gemini AI | Natural language summaries from run metrics |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `PyArrow` · `SQLite` · `Scikit-learn` · `XGBoost` · `Prophet` · `TensorFlow` · `Matplotlib` · `Plotly` · `DVC` · `MLflow` · `Gemini AI` · `FastAPI` · `Uvicorn` · `Docker` · `Pytest` · `Ruff` · `pre-commit` · `GitHub Actions`

---

## 📁 Project Structure

```
📦 demand-forecast-mlops
 ┣ 📂 .github/
 ┃ ┗ 📂 workflows/          # CI/CD pipelines
 ┣ 📂 data/                 # Raw and processed datasets (managed by DVC)
 ┣ 📂 notebooks/            # EDA and experimentation
 ┣ 📂 src/
 ┃ ┣ 📂 api/                # FastAPI endpoints
 ┃ ┣ 📂 features/           # Feature store and preprocessing
 ┃ ┗ 📂 models/             # Training scripts and MLflow integration
 ┣ 📂 tests/                # pytest test suite
 ┣ 📂 models/               # Serialized trained models
 ┣ 📂 mlruns/               # MLflow artifacts
 ┣ 📂 requirements          # Environment-specific dependencies (base, dev, test)
 ┣ 📜 run_pipeline.py       # Entrypoint: train, evaluate and register
 ┣ 📜 dvc.yaml              # DVC pipeline definition
 ┣ 📜 Dockerfile
 ┣ 📜 docker-compose.yml    # Orchestration: training job + MLflow UI
 ┣ 📜 pyproject.toml        # Ruff and tooling config
 ┗ 📜 requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root before running:

```
GOOGLE_API_KEY=your_api_key_here
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

> `GOOGLE_API_KEY` is required for the `/report` endpoint (Gemini AI). The rest of the pipeline runs without it.

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/ledesma-ivan/demand-forecast-mlops.git
cd demand-forecast-mlops

# 2. Pull versioned data
dvc pull

# 3. Start MLflow UI + training job
docker compose up --build
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |

To run only the training job:

```bash
docker compose run training_job
```

---

### Option B — Local

```bash
# 1. Clone the repo
git clone https://github.com/ledesma-ivan/demand-forecast-mlops.git
cd demand-forecast-mlops

# 2. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull versioned data
dvc pull

# 5. Run the training pipeline
python run_pipeline.py

# 6. Start the prediction API
uvicorn src.api.main:app --reload

# 7. Launch MLflow UI
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

| Service | URL |
|---------|-----|
| API | http://127.0.0.1:8000 |
| Swagger docs | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

---

## 🌐 API Usage

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

> 💡 Explore all endpoints interactively at **http://127.0.0.1:8000/docs** (FastAPI Swagger UI)

---

## 📈 Model Results

| Model   | RMSE      | MAPE | Training Time |
|---------|-----------|------|---------------|
| LSTM    | 1,098.77  | 5.1% | ~1m 10s       |
| XGBoost | 1,640.57  | 4.0% | ~11s          |
| Prophet | 2,313.03  | 9.4% | ~19s          |

> LSTM achieves the lowest RMSE while XGBoost leads on MAPE and training efficiency (~8x faster).
> For production use cases requiring frequent retraining, XGBoost offers the best accuracy/speed trade-off.
> All runs are tracked and compared via MLflow.

---

## 🔬 Technical Decisions

- **EDA:** Exploratory analysis on the Walmart dataset to identify weekly seasonality, holiday impact and sales variability across departments
- **Feature store:** Local implementation with Parquet (PyArrow) for feature persistence and reuse between runs
- **Feature engineering:** Temporal features (week, month, quarter, year-end), markdowns, sales lags (1, 2, 4, 8, 52 weeks), rolling statistics (mean, std, max), and cross-store/department features
- **Models evaluated:** XGBoost as baseline, Prophet for seasonality and trend, LSTM with TensorFlow for non-linear sequential patterns
- **Evaluation metrics:** RMSE and MAPE per model, automatically logged in MLflow for cross-run comparison
- **Storage:** SQLite for MLflow experiment persistence and data querying via SQL
- **Visualization:** Matplotlib for EDA in notebooks, Plotly for interactive model comparison vs real sales
- **LLM reporting:** Gemini AI (`gemini-3.1-flash-lite-preview`) to automatically generate natural language reports from MLflow run metrics

---

## ✅ Code Quality & Testing

This project uses industry-standard tooling for consistency and reliability:

| Tool | Purpose |
|------|---------|
| `pytest` | Unit and integration testing |
| `pytest-cov` | Test coverage reporting |
| `Ruff` | Fast linting and formatting (replaces flake8 + isort) |
| `pre-commit` | Automatic checks before every commit |
| `GitHub Actions` | CI runs on every push |

```bash
# Run tests
pytest

# Run tests with coverage report
pytest --cov=src tests/

# Lint and format
ruff check --fix .
ruff format .
```

---

## 🔄 CI/CD Pipeline

Automated pipeline via **GitHub Actions** triggered on every push to `main`:

| Stage | Tool | Description |
|-------|------|-------------|
| Code quality | Ruff + pre-commit | PEP8 style and import ordering |
| Testing | pytest + pytest-cov | Full test suite with coverage report |
| Build | Docker | Container image built and validated |
