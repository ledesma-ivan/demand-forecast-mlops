# tests/test_api.py
from fastapi.testclient import TestClient

from src.api.main import app  # Ajustá esto a cómo importes tu app de FastAPI

client = TestClient(app)


def test_predict_endpoint():
    """Prueba que el endpoint reciba bien el JSON y devuelva un 200 OK"""
    payload = {"store": 1, "dept": 1, "date": "2010-11-26"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    # Reemplazá "prediction" por la clave real que devuelve tu API
    assert "prediction" in response.json()


def test_predict_validation_error():
    """Prueba que FastAPI (Pydantic) bloquee requests mal formados"""
    # Falta el campo 'date'
    payload = {"store": 1, "dept": 1}
    response = client.post("/predict", json=payload)

    # 422 es el código de FastAPI para 'Unprocessable Entity' (error de validación)
    assert response.status_code == 422
