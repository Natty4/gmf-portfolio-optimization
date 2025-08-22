# src/models.py
import numpy as np
import pandas as pd
import math
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def fit_arima_on_log_returns(tsla_log_ret_train: pd.Series):
    """Fit ARIMA model on TSLA log returns."""
    model = auto_arima(
        tsla_log_ret_train.dropna(),
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=0,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    return model


def arima_predict_prices(
    model, tsla_log_ret_train: pd.Series, tsla_price_train_last: float, steps: int
):
    """Predict prices using ARIMA model."""
    fc_log_returns = model.predict(n_periods=steps)
    # Convert log-returns to price path
    prices = [tsla_price_train_last]
    for r in fc_log_returns:
        prices.append(prices[-1] * np.exp(r))
    prices = np.array(prices[1:])
    return fc_log_returns, prices


def make_lstm_sequences(series: pd.Series, lookback: int = 60):
    """Create sequences for LSTM training."""
    x, y = [], []
    vals = series.values.reshape(-1, 1)
    for i in range(lookback, len(vals)):
        x.append(vals[i - lookback : i, 0])
        y.append(vals[i, 0])

    x = np.array(x)
    y = np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    return x, y


def build_lstm(input_shape):
    """Build LSTM model architecture."""
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_lstm(
    train_series: pd.Series, val_split: float = 0.1, lookback: int = 60, seed: int = 42
):
    """Train LSTM model."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()

    x, y = make_lstm_sequences(pd.Series(scaled, index=train_series.index), lookback)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model = build_lstm(input_shape=(x.shape[1], x.shape[2]))
    history = model.fit(
        x,
        y,
        epochs=30,
        batch_size=32,
        validation_split=val_split,
        callbacks=[es],
        verbose=0,
    )
    return model, scaler


def lstm_recursive_forecast(
    model,
    scaler: StandardScaler,
    train_series: pd.Series,
    steps: int,
    lookback: int = 60,
):
    """Generate recursive forecasts using LSTM."""
    scaled_train = scaler.transform(train_series.values.reshape(-1, 1)).flatten()
    window = list(scaled_train[-lookback:])
    preds_scaled = []
    for _ in range(steps):
        x = np.array(window).reshape((1, lookback, 1))
        yhat_scaled = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(yhat_scaled)
        window.pop(0)
        window.append(yhat_scaled)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


def calculate_metrics(y_true, y_pred):
    """Calculate model evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = (
        np.abs((y_true - y_pred) / y_true).replace([np.inf, -np.inf], np.nan).dropna()
    ).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}
