import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred):
    """Calcula y retorna RMSE y MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

def plot_model_comparison(dates, y_true, preds_dict, title="Comparación de Predicciones"):
    """Dibuja el gráfico comparativo final."""
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true.values, label='Ventas Reales', color='black', linewidth=2, marker='o')
    
    styles = ['--', '-.', ':']
    for (name, preds), style in zip(preds_dict.items(), styles):
        plt.plot(dates, preds, label=name, linestyle=style)
        
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Ventas Semanales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()