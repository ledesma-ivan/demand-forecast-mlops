import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, model_name="Modelo"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"--- Resultados {model_name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.4f}")
    return rmse, mape

def plot_predictions(dates, y_true, y_pred, title="Predicciones vs Reales"):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label='Ventas Reales', color='black', marker='o')
    plt.plot(dates, y_pred, label='Predicción XGBoost', linestyle='--')
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Ventas Semanales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()