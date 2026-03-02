import os
import pandas as pd
from src.features.build_features import clean_and_merge_data, create_features
from src.features.feature_store import LocalFeatureStore
from src.models.train_xgboost import temporal_train_test_split, train_model, save_model
from src.evaluation.metrics import evaluate_model, plot_predictions
import time

def main():
    print("--- INICIANDO PIPELINE DE WALMART FORECASTING ---")
    
    # 1. Carga de datos crudos (Asegúrate de que las rutas sean correctas)
    print("Cargando datos crudos...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    features_df = pd.read_csv("data/features.csv")
    stores_df = pd.read_csv("data/stores.csv")
    
    # 2. Limpieza y Feature Engineering
    print("Ejecutando Feature Engineering...")
    df_merged = clean_and_merge_data(train_df, features_df, stores_df)
    
    fs = LocalFeatureStore(base_dir='data/processed/walmart_features')
    
    try:
        # Intentamos cargar la versión 2 (no existe, así que pasará al except)
        df_model = fs.load_features('master_features_v2')
    except FileNotFoundError:
        # Como no existe, lo calcula y lo guarda como versión 2
        df_model = create_features(df_merged)
        fs.save_features(df_model, 'master_features_v2')
        
    # 3. Preparación de datos para el modelo (Ejemplo: Tienda 1, Dept 1)
    print("Preparando datos para Entrenamiento...")
    X_train, X_test, y_train, y_test, test_dates = temporal_train_test_split(
        df=df_model, 
        split_date='2012-08-01', 
        store=1, 
        dept=1
    )
    
    # 4. Entrenamiento del Modelo
    print("Entrenando XGBoost...")
    model = train_model(X_train, y_train)
    
    # Guardar el modelo en disco (asegurando que la carpeta exista)
    os.makedirs("models", exist_ok=True)
    save_model(model, filepath="models/xgb_model.json")
    print("✅ Modelo guardado en models/xgb_model.json")
    
    # 5. Predicción y Evaluación
    print("Evaluando el modelo...")
    predictions = model.predict(X_test)
    
    rmse, mape = evaluate_model(y_test, predictions, model_name="XGBoost (Store 1, Dept 1)")
    
    # Generar gráfico
    plot_predictions(test_dates, y_test, predictions, title="Predicciones XGBoost - Store 1, Dept 1")
    
    print("--- PIPELINE FINALIZADO ---")

if __name__ == "__main__":
    main()