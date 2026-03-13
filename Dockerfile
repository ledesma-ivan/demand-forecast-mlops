# 1. Imagen base
FROM python:3.10-slim

# 2. Instalar dependencias del sistema operativo (necesario para XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copiar requirements
COPY requirements.txt .

# 4. Instalar librerías de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Comando por defecto (será sobrescrito por docker-compose en algunos casos)
CMD ["python", "run_pipeline.py"]
