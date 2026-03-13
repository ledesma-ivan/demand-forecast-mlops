from contextlib import asynccontextmanager
from datetime import datetime

import mlflow.pyfunc
import numpy as np  # <-- ¡FALTABA ESTO!
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.feature_store import LocalFeatureStore

# ==========================================
# 0. CONFIGURACIÓN DEL MODELO MANUAL
# ==========================================
RUN_ID = "6ff76bd98c8d44d39b92db40009c82ea"
ARTIFACT_PATH = "xgboost_model"

# Variables globales
MODEL = None
DF_FEATURES = None  # <-- Declarar aquí para evitar errores

MODEL_INFO = {"run_id": RUN_ID, "artifact_path": ARTIFACT_PATH, "loaded_at": None}


# ==========================================
# 1. CARGA DE MODELO Y DATOS (Al arrancar)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, DF_FEATURES
    print("⏳ Iniciando API...")

    # --- A) CARGAR MODELO ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
    try:
        MODEL = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Modelo cargado (Run ID: {RUN_ID})")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")

    # --- B) CARGAR DATOS REALES (Feature Store) ---
    try:
        # Asegúrate de que la ruta base coincida con tu proyecto
        fs = LocalFeatureStore(base_dir="data/processed/walmart_features")
        DF_FEATURES = fs.load_features("master_features_v2")

        # Convertimos la columna Date a string para facilitar la búsqueda
        DF_FEATURES["Date"] = pd.to_datetime(DF_FEATURES["Date"]).dt.strftime(
            "%Y-%m-%d"
        )
        print(f"✅ Feature Store cargado: {len(DF_FEATURES)} registros disponibles.")
    except Exception as e:
        print(f"❌ Error al cargar Feature Store: {e}")

    yield
    print("🛑 Apagando API...")


app = FastAPI(title="Walmart Forecast API (Datos Reales)", lifespan=lifespan)


# ==========================================
# 2. ESQUEMAS DE PETICIÓN
# ==========================================
class PredictRequest(BaseModel):
    store: int
    dept: int
    date: str  # Formato YYYY-MM-DD


# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.post("/predict")
def predict_sales(request: PredictRequest):
    if MODEL is None or DF_FEATURES is None:
        raise HTTPException(
            status_code=503, detail="Servicio no disponible temporalmente."
        )

    try:
        # 1. Validar fecha
        try:
            datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Formato de fecha inválido. Usa YYYY-MM-DD."
            )

        # 2. BUSCAR LOS DATOS REALES en el DataFrame cargado
        filtro = (
            (DF_FEATURES["Store"] == request.store)
            & (DF_FEATURES["Dept"] == request.dept)
            & (DF_FEATURES["Date"] == request.date)
        )

        fila_real = DF_FEATURES[filtro].copy()

        # Si no hay datos para esa combinación, devolvemos un error 404
        if fila_real.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron features para Store {request.store}, Dept {request.dept} en {request.date}.",
            )

        # 3. PREPARAR LA FILA PARA EL MODELO
        columnas_a_quitar = ["Date", "Weekly_Sales", "Store", "Dept"]
        for col in columnas_a_quitar:
            if col in fila_real.columns:
                fila_real = fila_real.drop(columns=[col])

        # Convertir booleanos a enteros
        for col in fila_real.columns:
            if fila_real[col].dtype == bool:
                fila_real[col] = fila_real[col].astype(int)

        # Quitar cualquier otra columna de texto sobrante
        columnas_texto = fila_real.select_dtypes(include=["object"]).columns
        if len(columnas_texto) > 0:
            fila_real = fila_real.drop(columns=columnas_texto)

        fila_real = fila_real.astype(float)

        # =========================================================
        # 🚨 AQUÍ IMPRIMIMOS LAS COLUMNAS ANTES DE QUE FALLE
        # =========================================================
        print("\n" + "=" * 50)
        print(f"📊 CANTIDAD DE COLUMNAS ACTUALES: {len(fila_real.columns)}")
        print(f"📋 LISTA DE COLUMNAS: {fila_real.columns.tolist()}")

        # --- MAGIA DE MLFLOW: ALINEAR COLUMNAS EXACTAS ---
        if MODEL.metadata.signature is not None:
            expected_cols = [col.name for col in MODEL.metadata.signature.inputs]
            print(f"🎯 COLUMNAS QUE ESPERA EL MODELO: {len(expected_cols)}")
            # Si el modelo guardó qué columnas esperaba, filtramos la fila para que encaje
            fila_real = fila_real[expected_cols]

        print("=" * 50 + "\n")

        # 4. PREDECIR
        prediction = MODEL.predict(fila_real)
        predicted_value = float(np.squeeze(prediction))

        return {
            "store": request.store,
            "dept": request.dept,
            "date": request.date,
            "predicted_weekly_sales": round(predicted_value, 2),
        }

    except HTTPException:
        # Si es un error 400, 404, o 503 que ya lanzamos nosotros, lo dejamos pasar
        raise
    except Exception as e:
        # SI ALGO FALLA, este bloque lo atrapa e imprime el error exacto
        print(f"❌ Error interno durante predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno general: {str(e)}")
