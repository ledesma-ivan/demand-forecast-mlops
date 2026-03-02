import os
import pandas as pd
from src.features.build_features import clean_and_merge_data, create_features
from src.features.feature_store import LocalFeatureStore
from src.evaluation.metrics import evaluate_model, plot_model_comparison

# Importar Modelos
from src.models.train_xgboost import temporal_train_test_split, train_predict_xgboost, save_model
from src.models.train_prophet import train_predict_prophet
from src.models.train_lstm import train_predict_lstm

def main():
    print("--- INICIANDO TORNEO DE MODELOS WALMART ---")
    
    # 1. Feature Store / Engineering
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
        
    # 2. Separación de datos
    train, test = temporal_train_test_split(df_model, split_date='2012-08-01', store=1, dept=1)
    y_test = test['Weekly_Sales']
    
    # 3. Entrenamiento y Recolección de Predicciones
    resultados = []
    diccionario_predicciones = {}
    
    # -- XGBoost --
    preds_xgb, time_xgb, modelo_xgb = train_predict_xgboost(train, test)
    rmse_xgb, mape_xgb = evaluate_model(y_test, preds_xgb)
    resultados.append({'Modelo': 'XGBoost', 'RMSE': rmse_xgb, 'MAPE': mape_xgb, 'Tiempo (s)': time_xgb})
    diccionario_predicciones['XGBoost'] = preds_xgb
    
    # -- Prophet --
    preds_prophet, time_prophet = train_predict_prophet(train, test)
    rmse_prophet, mape_prophet = evaluate_model(y_test, preds_prophet)
    resultados.append({'Modelo': 'Prophet', 'RMSE': rmse_prophet, 'MAPE': mape_prophet, 'Tiempo (s)': time_prophet})
    diccionario_predicciones['Prophet'] = preds_prophet
    
    # -- LSTM --
    preds_lstm, time_lstm = train_predict_lstm(train, test)
    rmse_lstm, mape_lstm = evaluate_model(y_test, preds_lstm)
    resultados.append({'Modelo': 'LSTM', 'RMSE': rmse_lstm, 'MAPE': mape_lstm, 'Tiempo (s)': time_lstm})
    diccionario_predicciones['LSTM'] = preds_lstm

    # 4. Tabla Comparativa Final
    df_resultados = pd.DataFrame(resultados).set_index('Modelo')
    print("\n=== TABLA COMPARATIVA DE MODELOS ===")
    print(df_resultados.sort_values('RMSE'))
    
    # 5. Guardar el modelo ganador (XGBoost) para producción
    os.makedirs("models", exist_ok=True)
    save_model(modelo_xgb, filepath="models/xgb_model.json")
    print("\n✅ Modelo ganador (XGBoost) guardado en models/xgb_model.json")
    
    # 6. Mostrar el gráfico
    plot_model_comparison(test['Date'], y_test, diccionario_predicciones, "Batalla de Modelos: Tienda 1, Dept 1")

if __name__ == "__main__":
    main()