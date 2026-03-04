import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import mlflow.prophet       # Añadido para Prophet
import mlflow.tensorflow    # Añadido para LSTM (Keras)

from src.features.build_features import clean_and_merge_data, create_features
from src.features.feature_store import LocalFeatureStore
from src.evaluation.metrics import evaluate_model, plot_model_comparison
from src.models.train_xgboost import temporal_train_test_split, train_predict_xgboost
from src.models.train_prophet import train_predict_prophet
from src.models.train_lstm import train_predict_lstm

def main():
    print("--- INICIANDO TORNEO DE MODELOS CON MLFLOW ---")
    
    # NUEVO: Esperar 10 segundos para que el servidor de MLflow inicie en Docker
    print("Esperando 10 segundos a que el servidor MLflow esté listo...")
    time.sleep(10)

    # 1. Configurar MLflow Tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("Walmart_Demand_Forecasting")
    
    # 2. Feature Store / Engineering
    fs = LocalFeatureStore(base_dir='data/processed/walmart_features')
    try:
        df_model = fs.load_features('master_features_v2')
    except FileNotFoundError:
        print("Calculando features desde cero...")
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        features_df = pd.read_csv("data/features.csv")
        stores_df = pd.read_csv("data/stores.csv")
        
        df_merged = clean_and_merge_data(train_df, features_df, stores_df)
        df_model = create_features(df_merged)
        fs.save_features(df_model, 'master_features_v2')
        
    # 3. Separación de datos
    train, test = temporal_train_test_split(df_model, split_date='2012-08-01', store=1, dept=1)
    y_test = test['Weekly_Sales']
    
    resultados = []
    diccionario_predicciones = {}
    
    # ==========================================
    # ENTRENAMIENTO XGBOOST
    # ==========================================
    print("\n--- Entrenando XGBoost ---")
    mlflow.xgboost.autolog() 
    
    with mlflow.start_run(run_name="XGBoost_Baseline"):
        mlflow.set_tag("Store", "1")
        mlflow.set_tag("Dept", "1")
        mlflow.log_param("Split_Date", "2012-08-01")
        
        preds_xgb, time_xgb, modelo_xgb = train_predict_xgboost(train, test)
        rmse_xgb, mape_xgb = evaluate_model(y_test, preds_xgb)
        
        mlflow.log_metric("val_rmse", rmse_xgb)
        mlflow.log_metric("val_mape", mape_xgb)
        mlflow.log_metric("training_time_seconds", time_xgb)
        
        mlflow.xgboost.log_model(modelo_xgb, artifact_path="xgboost_model")
        
        resultados.append({'Modelo': 'XGBoost', 'RMSE': rmse_xgb, 'MAPE': mape_xgb})
        diccionario_predicciones['XGBoost'] = preds_xgb

    # ==========================================
    # ENTRENAMIENTO PROPHET
    # ==========================================
    print("\n--- Entrenando Prophet ---")
    with mlflow.start_run(run_name="Prophet_Baseline"):
        mlflow.set_tag("Store", "1")
        mlflow.set_tag("Dept", "1")
        mlflow.log_param("Split_Date", "2012-08-01")
        
        preds_prophet, time_prophet, modelo_prophet = train_predict_prophet(train, test)
        rmse_prophet, mape_prophet = evaluate_model(y_test, preds_prophet)
        
        mlflow.log_metric("val_rmse", rmse_prophet)
        mlflow.log_metric("val_mape", mape_prophet)
        mlflow.log_metric("training_time_seconds", time_prophet)
        
        # Guardamos el modelo en MLflow
        mlflow.prophet.log_model(modelo_prophet, artifact_path="prophet_model")
        
        resultados.append({'Modelo': 'Prophet', 'RMSE': rmse_prophet, 'MAPE': mape_prophet})
        diccionario_predicciones['Prophet'] = preds_prophet
    
    # ==========================================
    # ENTRENAMIENTO LSTM
    # ==========================================
    print("\n--- Entrenando LSTM ---")
    mlflow.tensorflow.autolog() # Autologging para Keras/Tensorflow
    
    with mlflow.start_run(run_name="LSTM_Baseline"):
        mlflow.set_tag("Store", "1")
        mlflow.set_tag("Dept", "1")
        mlflow.log_param("Split_Date", "2012-08-01")
        
        preds_lstm, time_lstm, modelo_lstm = train_predict_lstm(train, test)
        rmse_lstm, mape_lstm = evaluate_model(y_test, preds_lstm)
        
        mlflow.log_metric("val_rmse", rmse_lstm)
        mlflow.log_metric("val_mape", mape_lstm)
        mlflow.log_metric("training_time_seconds", time_lstm)
        
        # Guardamos el modelo en MLflow
        mlflow.tensorflow.log_model(modelo_lstm, artifact_path="lstm_model")
        
        resultados.append({'Modelo': 'LSTM', 'RMSE': rmse_lstm, 'MAPE': mape_lstm})
        diccionario_predicciones['LSTM'] = preds_lstm

    # 4. Tabla Comparativa Final
    df_resultados = pd.DataFrame(resultados).set_index('Modelo')
    print("\n=== TABLA COMPARATIVA DE MODELOS ===")
    print(df_resultados.sort_values('RMSE'))
    
    # 5. Mostrar el gráfico
    plot_model_comparison(test['Date'], y_test, diccionario_predicciones, "Batalla de Modelos: Tienda 1, Dept 1")
    print("\n✅ Pipeline ejecutado. Abre MLflow UI para ver los resultados.")

if __name__ == "__main__":
    main()