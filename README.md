# Demand Forecast MLOps

Pipeline de ML de punta a punta para forecasting de demanda retail — desde los datos crudos hasta una API REST lista para producción, con tracking completo de experimentos via MLflow.

---

## Stack tecnológico

`Python` · `Pandas` · `NumPy`· `SQLite` · `Scikit-learn` · `XGBoost` · `TensorFlow` · `Matplotlib` · `Plotly` · `MLflow` · `FastAPI` · `Uvicorn` · `Docker`

---

## ¿Qué hace?

- Entrena y compara múltiples modelos de forecasting (XGBoost, TensorFlow) sobre datos históricos de retail (tienda, departamento, fecha)
- Aplica feature engineering sobre variables temporales y de contexto de negocio
- Registra cada experimento (parámetros, métricas, artefactos) con **MLflow** para gestión del ciclo de vida del modelo
- Expone predicciones a través de una API REST con **FastAPI**
- Orquesta el entrenamiento y la UI de MLflow mediante **Docker Compose**

---

## Decisiones técnicas

- **EDA:** análisis exploratorio sobre el dataset de Walmart para identificar estacionalidad semanal, impacto de feriados y variabilidad por departamento
- **Feature engineering:** extracción de variables temporales (semana, mes, trimestre), lags de ventas y codificación de tienda/departamento
- **Modelos evaluados:**  XGBoost como baseline, Prophet para capturar estacionalidad y tendencia, y red neuronal con TensorFlow para patrones no lineales
- **Métricas de evaluación:** RMSE y MAPE por modelo, registradas automáticamente en MLflow para comparación entre runs
- **Almacenamiento:**: Base de datos SQLite para persistencia de experimentos MLflow y consulta de datos via SQL
- **Visualización:** Visualización: Matplotlib para análisis exploratorio en notebooks, Plotly para gráficos interactivos de comparación entre XGBoost, Prophet y TensorFlow vs ventas reales


---

## Estructura del proyecto

```
├── data/                # Datasets crudos y procesados
├── notebooks/           # EDA y experimentación
├── src/                 # Código fuente del pipeline y la API
│   └── api/             # Aplicación FastAPI
├── models/              # Modelos entrenados serializados
├── mlruns/              # Artefactos de MLflow
├── Dockerfile           # Imagen base del proyecto
├── docker-compose.yml   # Orquestación: training job + MLflow UI
├── run_pipeline.py      # Punto de entrada: entrenar, evaluar y registrar
└── requirements.txt
```

---

## Inicio rápido

### Opción A — Con Docker (recomendado)

```bash
# Levantar MLflow UI + job de entrenamiento
docker compose up --build
```

| Servicio | URL |
|----------|-----|
| MLflow UI | http://localhost:5000 |

Para correr solo el entrenamiento:

```bash
docker compose run training_job
```

### Opción B — Local

```bash
# 1. Crear y activar entorno virtual
python -m venv venv && source venv/bin/activate  # Mac/Linux
python -m venv venv && venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el pipeline de entrenamiento
python run_pipeline.py

# 4. Levantar la API de predicción
uvicorn src.api.main:app --reload

# 5. Abrir la interfaz de MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

| Servicio | URL |
|----------|-----|
| API | http://127.0.0.1:8000 |
| Documentación (Swagger) | http://127.0.0.1:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

---

## Uso de la API

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
