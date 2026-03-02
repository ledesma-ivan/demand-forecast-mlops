import xgboost as xgb
import pandas as pd
import time

def temporal_train_test_split(df, split_date='2012-08-01', store=1, dept=1):
    """Filtra y divide los datos devolviendo train y test completos."""
    df_subset = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
    df_subset['Date'] = pd.to_datetime(df_subset['Date'])
    df_subset = df_subset.sort_values('Date')
    
    train = df_subset[df_subset['Date'] < split_date].copy()
    test = df_subset[df_subset['Date'] >= split_date].copy()
    
    return train, test

def train_predict_xgboost(train, test):
    print("--- Entrenando XGBoost ---")
    start_time = time.time()
    
    drop_cols = ['Date', 'Weekly_Sales', 'Store', 'Dept', 'Type']
    X_train = train.drop(columns=drop_cols)
    y_train = train['Weekly_Sales']
    X_test = test.drop(columns=drop_cols)
    
    modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X_train, y_train)
    
    preds = modelo.predict(X_test)
    xgb_time = time.time() - start_time
    print("✅ XGBoost Listo!")
    
    # Retornamos las predicciones, el tiempo y el modelo guardable
    return preds, xgb_time, modelo

def save_model(model, filepath='models/xgb_model.json'):
    model.save_model(filepath)