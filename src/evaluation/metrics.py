import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def evaluate_model(y_true, y_pred):
    """Calcula y retorna RMSE y MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape


def plot_model_comparison(dates, y_true, preds_dict, title="Comparación de Predicciones"):
    """Dibuja el gráfico comparativo final con Plotly."""
    fig = go.Figure()

    # Ventas reales
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_true.values,
            name="Ventas Reales",
            line=dict(color="black", width=2),
            mode="lines+markers",
        )
    )

    # Predicciones por modelo
    styles = ["dash", "dashdot", "dot"]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for (name, preds), style, color in zip(preds_dict.items(), styles, colors):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=preds,
                name=name,
                line=dict(dash=style, color=color),
                mode="lines",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Ventas Semanales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
        width=1000,
        height=500,
    )

    fig.show()
