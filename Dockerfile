# Usamos una imagen oficial de Python
FROM python:3.10-slim

# Instalamos dependencias del sistema necesarias para compilar Prophet, Pandas y C++ extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el requirements y lo instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Comando por defecto (será sobrescrito por docker-compose en algunos casos)
CMD ["python", "run_pipeline.py"]
