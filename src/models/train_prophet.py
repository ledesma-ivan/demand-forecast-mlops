import time

from prophet import Prophet


def train_predict_prophet(train, test):
    print("--- Entrenando Prophet ---")
    start_time = time.time()

    train_prophet = train[["Date", "Weekly_Sales", "IsHoliday"]].rename(
        columns={"Date": "ds", "Weekly_Sales": "y"}
    )
    test_prophet = test[["Date", "IsHoliday"]].rename(columns={"Date": "ds"})

    modelo = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    modelo.add_regressor("IsHoliday")
    modelo.fit(train_prophet)

    preds_df = modelo.predict(test_prophet)
    preds = preds_df["yhat"].values
    prophet_time = time.time() - start_time

    print("✅ Prophet Listo!")
    # Retornamos las predicciones, el tiempo y el modelo
    return preds, prophet_time, modelo
