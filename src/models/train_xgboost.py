import xgboost as xgb
import pandas as pd
import time

def temporal_train_test_split(df, split_date='2012-08-01', store=None, dept=None):
    """Filtra y divide los datos en el tiempo."""
    # 1. Filtramos por tienda y departamento
    if store and dept:
        df = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 2. Hacemos el split temporal (aquí es donde NACEN train y test)
    train = df[df['Date'] < split_date].copy()
    test = df[df['Date'] >= split_date].copy()
    
    # 3. Separamos X (features) e y (target)
    drop_cols = ['Date', 'Weekly_Sales', 'Store', 'Dept', 'Type']
    X_train = train.drop(columns=drop_cols)
    y_train = train['Weekly_Sales']
    X_test = test.drop(columns=drop_cols)
    y_test = test['Weekly_Sales']
    
    return X_train, X_test, y_train, y_test, test['Date']

def train_model(X_train, y_train, params=None):
    """Entrena y retorna el modelo XGBoost."""
    print("--- Entrenando XGBoost ---")
    start_time = time.time()
    
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
    
    # XGBoost maneja los NaNs internamente
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    xgb_time = time.time() - start_time
    print(f"✅ XGBoost Listo en {xgb_time:.2f} segundos!")
    
    return model

def save_model(model, filepath='models/xgb_model.json'):
    """Guarda el modelo para ser consumido luego por la API."""
    model.save_model(filepath)