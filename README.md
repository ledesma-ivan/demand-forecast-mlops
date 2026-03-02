## Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar el pipeline

```bash
python run_pipeline.py
```

---

## Levantar la interfaz gráfica de MLflow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Luego abre en tu navegador:

```
http://127.0.0.1:5000
```

---

## Desactivar el entorno virtual

```bash
deactivate
```

---
