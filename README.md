# Demand Forecast MLOps

End-to-end ML pipeline for retail demand forecasting — from raw data to a production-ready REST API, with full experiment tracking via MLflow.

---

## Tech Stack

`Python` · `Scikit-learn` · `XGBoost` · `Pandas` · `NumPy` · `MLflow` · `FastAPI` · `Uvicorn`

---

## What it does

- Trains a demand forecasting model on historical retail data (store, department, date)
- Tracks every experiment (parameters, metrics, artifacts) with **MLflow**
- Exposes predictions through a **FastAPI** REST API

---

## Project Structure

```
├── data/            # Raw and processed datasets
├── notebooks/       # EDA and experimentation
├── src/             # Pipeline and API source code
│   └── api/         # FastAPI app
├── models/          # Serialized trained models
├── mlruns/          # MLflow artifacts
├── run_pipeline.py  # Entrypoint: train + evaluate + register
└── requirements.txt
```

---

## Quickstart

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
