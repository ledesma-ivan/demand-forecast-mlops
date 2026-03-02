import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_predict_lstm(train, test):
    print("--- Entrenando LSTM ---")
    tf.random.set_seed(42)
    start_time = time.time()
    
    drop_cols = ['Date', 'Weekly_Sales', 'Store', 'Dept', 'Type']
    X_train_lstm = train.drop(columns=drop_cols).fillna(0)
    X_test_lstm = test.drop(columns=drop_cols).fillna(0)
    y_train = train['Weekly_Sales']

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_lstm)
    X_test_scaled = scaler_x.transform(X_test_lstm)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    modelo = Sequential()
    modelo.add(LSTM(50, activation='relu', input_shape=(X_train_3d.shape[1], X_train_3d.shape[2])))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mse')

    modelo.fit(X_train_3d, y_train_scaled, epochs=50, batch_size=16, verbose=0)

    preds_scaled = modelo.predict(X_test_3d, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    lstm_time = time.time() - start_time

    print("✅ LSTM Listo!")
    return preds, lstm_time