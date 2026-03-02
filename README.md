## Crear entorno virtual

### Windows

```bash
python3 -m venv venv
venv\Scripts\activate
````

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar pipeline

```bash
python run_pipeline.py
```

```
## Levantar interfaz grafica de ml flow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Desactivar entorno

```bash
deactivate
```

```