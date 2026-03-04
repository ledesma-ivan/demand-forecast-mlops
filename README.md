# Demand Forecast MLOps

End-to-end ML pipeline for retail demand forecasting — from raw data to a production-ready REST API, with full experiment tracking via MLflow.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `SQLite` · `PyArrow` · `Scikit-learn` · `XGBoost` · `Prophet` · `TensorFlow` · `Matplotlib` · `Plotly` · `MLflow` · `Gemini AI` · `FastAPI` · `Uvicorn` · `Docker` · `AWS`

---

## What it does

- Trains and compares multiple forecasting models (XGBoost, Prophet, LSTM) on historical retail data (store, department, date)
- Implements a local feature store with Parquet for feature persistence and reuse across runs
- Applies advanced feature engineering on temporal variables, markdowns, lags and cross-series features
- Tracks every experiment (parameters, metrics, artifacts) with **MLflow** for full model lifecycle management
- Generates natural language reports from MLflow metrics via **Gemini AI (gemini-3.1-flash-lite-preview)**
- Exposes predictions through a REST API built with **FastAPI**
- Orchestrates the training job and MLflow UI via **Docker Compose**
- Artifacts stored in **AWS S3**, API deployed on **AWS App Runner**

---

## Results

| Model | RMSE | MAPE |
|-------|------|------|
| LSTM | 1,098.77 | 5.1% |
| XGBoost | 1,640.57 | 4.0% |
| Prophet | 2,313.03 | 9.4% |

> LSTM achieves the lowest RMSE while XGBoost leads on MAPE — trade-off between overall error and percentage accuracy tracked via MLflow across all runs.

---

## Technical Decisions

- **EDA:** exploratory analysis on the Walmart dataset to identify weekly seasonality, holiday impact and sales variability across departments
- **Feature store:** local implementation with Parquet (PyArrow) for feature persistence and reuse between runs
- **Feature engineering:** temporal features (week, month, quarter, year-end), markdowns, sales lags (1, 2, 4, 8, 52 weeks), rolling statistics (mean, std, max) and cross-store/department features
- **Models evaluated:** XGBoost as baseline, Prophet for seasonality and trend, and LSTM with TensorFlow for non-linear sequential patterns
- **Evaluation metrics:** RMSE and MAPE per model, automatically logged in MLflow for cross-run comparison
- **Storage:** SQLite for MLflow experiment persistence and data querying via SQL
- **Visualization:** Matplotlib for EDA in notebooks, Plotly for interactive model comparison vs real sales
- **LLM reporting:** Gemini AI (gemini-3.1-flash-lite-preview) to automatically generate natural language reports from MLflow run metrics
- **Cloud:** artifacts stored in AWS S3, API deployed on App Runner



---

## Project Structure

```
├── data/                # Raw and processed datasets
├── notebooks/           # EDA and experimentation
├── src/                 # Pipeline and API source code
│   └── api/             # FastAPI app
├── models/              # Serialized trained models
├── mlruns/              # MLflow artifacts
├── Dockerfile           # Base image
├── docker-compose.yml   # Orchestration: training job + MLflow UI
├── run_pipeline.py      # Entrypoint: train, evaluate and register
└── requirements.txt
```

---

## Quickstart

### Option A — With Docker (recommended)

```bash
# Start MLflow UI + training job
docker compose up --build
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |

To run only the training job:

```bash
docker compose run training_job
```

### Option B — Local

```bash
# 1. Create and activate virtual environment
python -m venv venv && source venv/bin/activate  # Mac/Linux
python -m venv venv && venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training pipeline
python run_pipeline.py

# 4. Start the prediction API
uvicorn src.api.main:app --reload

# 5. Launch MLflow UI
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

| Service | URL |
|---------|-----|
| API | http://127.0.0.1:8000 |
| Swagger docs | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

---

## API Usage

`POST /predict`

```json
{
  "store": 1,
  "dept": 1,
  "date": "2010-11-26"
}
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"store": 1, "dept": 1, "date": "2010-11-26"}'
```

`GET /report` — generates a natural language summary of the latest MLflow run via Groq

```bash
curl http://127.0.0.1:8000/report
```
