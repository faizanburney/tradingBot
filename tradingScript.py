import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# ---------------------------
# 1. Data Fetching
# ---------------------------
def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical OHLCV data for a given symbol using yfinance.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

# ---------------------------
# 2. Feature Engineering
# ---------------------------
def feature_engineering(data):
    """
    Create technical indicators and candlestick pattern features.
    """
    # Simple Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Daily Return and Volatility
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=10).std()

    # --- RSI Calculation ---
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)  # avoid division by zero
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # --- MACD Calculation ---
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # --- Candlestick Pattern: Bullish Engulfing ---
    # Definition:
    #   - Yesterday's candle is bearish (Close < Open)
    #   - Today's candle is bullish (Close > Open)
    #   - Today's Open is lower than yesterday's Close and today's Close is higher than yesterday's Open.
    bullish = []
    for i in range(len(data)):
        if i == 0:
            bullish.append(0)
        else:
            prev = data.iloc[i - 1]
            curr = data.iloc[i]
            if (prev['Close'].item() < prev['Open'].item() and
                curr['Close'].item() > curr['Open'].item() and
                curr['Open'].item() < prev['Close'].item() and
                curr['Close'].item() > prev['Open'].item()):
                bullish.append(1)
            else:
                bullish.append(0)
    data['Bullish_Engulfing'] = bullish

    # --- Candlestick Pattern: Bearish Engulfing ---
    # Definition:
    #   - Yesterday's candle is bullish (Close > Open)
    #   - Today's candle is bearish (Close < Open)
    #   - Today's Open is higher than yesterday's Close and today's Close is lower than yesterday's Open.
    bearish = []
    for i in range(len(data)):
        if i == 0:
            bearish.append(0)
        else:
            prev = data.iloc[i - 1]
            curr = data.iloc[i]
            if (prev['Close'].item() > prev['Open'].item() and
                curr['Close'].item() < curr['Open'].item() and
                curr['Open'].item() > prev['Close'].item() and
                curr['Close'].item() < prev['Open'].item()):
                bearish.append(1)
            else:
                bearish.append(0)
    data['Bearish_Engulfing'] = bearish

    # Drop rows with NaN values generated from rolling calculations
    data.dropna(inplace=True)
    return data

# ---------------------------
# 3. Create Sequences for the Deep Learning Model
# ---------------------------
def create_sequences(data, window_size, target_col='Close'):
    """
    Create sliding-window sequences. Each sample is a window of days with features,
    and the target is the closing price immediately after the window.
    """
    # Define the feature columns: include raw OHLCV plus the technical indicators and pattern flags.
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_10', 'SMA_50', 'Return', 'Volatility', 
                    'RSI', 'MACD', 'MACD_Signal', 
                    'Bullish_Engulfing', 'Bearish_Engulfing']
    data_features = data[feature_cols].values
    data_target = data[target_col].values

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data_features[i-window_size:i])
        y.append(data_target[i])
    return np.array(X), np.array(y)

# ---------------------------
# 4. Build a Robust Deep Learning Model (Hybrid CNN + LSTM)
# ---------------------------
def build_model(input_shape):
    """
    Build a hybrid CNN-LSTM model that learns local patterns (via CNN layers) and 
    sequential dependencies (via LSTM layers).
    """
    model = Sequential()
    # Convolutional layer to extract local features
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # LSTM layer to capture sequential patterns over the window
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers for prediction
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))  # Regression output: next day's closing price
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()
    return model

# ---------------------------
# 5. Train the Model
# ---------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the model using training data and validate on a separate set.
    """
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)
    return history

# ---------------------------
# 6. Trading Decision Logic
# ---------------------------
def make_trading_decision(model, latest_sequence, current_price, threshold=0.01):
    """
    Make a simple trading decision:
      - "Buy" if the predicted price exceeds the current price by more than the threshold.
      - "Sell" if it is lower by more than the threshold.
      - "Hold" otherwise.
    """
    predicted_price = model.predict(latest_sequence)[0][0]
    
    if predicted_price > current_price * (1 + threshold):
        decision = "Buy"
    elif predicted_price < current_price * (1 - threshold):
        decision = "Sell"
    else:
        decision = "Hold"
        
    return decision, predicted_price

# ---------------------------
# Main Pipeline
# ---------------------------
if __name__ == "__main__":
    # Parameters
    symbol = "AAPL"  # Replace with desired ticker (or extend for crypto using ccxt)
    start_date = "2018-01-01"
    end_date = "2024-01-01"
    window_size = 60  # Sequence length (e.g., 60 days)

    # 1. Fetch and prepare the data
    print("Fetching data...")
    data = fetch_data(symbol, start_date, end_date)
    data = feature_engineering(data)
    
    # 2. Create sequences from the data
    print("Creating sequences...")
    X, y = create_sequences(data, window_size, target_col='Close')
    
    # 3. Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Normalize features (reshape to 2D for the scaler, then back to original shape)
    num_features = X.shape[2]
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)
    
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    
    # 5. Build and train the model
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = build_model(input_shape)
    print("Training the model...")
    train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=50, batch_size=32)
    
    # 6. Make a trading decision based on the latest available sequence
    latest_sequence = X_val_scaled[-1:]  # Use the last window from the validation set
    current_price = data['Close'].iloc[-1].item()
    decision, predicted_price = make_trading_decision(model, latest_sequence, current_price, threshold=0.01)
    
    print(f"\nCurrent Price: {current_price:.2f}")
    print(f"Predicted Next Price: {predicted_price:.2f}")
    print(f"Trading Decision: {decision}")
