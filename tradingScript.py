import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Define a custom Attention layer
# -------------------------------
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal", 
                                 trainable=True)
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[-1],),
                                 initializer="zeros", 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores for each time step and generate a context vector.
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

# -------------------------------
# 2. Helper functions for chart pattern detection
# -------------------------------
def local_maxima(prices):
    """Return indices of local maxima in a 1D array."""
    maxima = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            maxima.append(i)
    return maxima

def local_minima(prices):
    """Return indices of local minima in a 1D array."""
    minima = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            minima.append(i)
    return minima

def detect_head_and_shoulders(prices, tolerance=0.03):
    """Detect a simple head and shoulders pattern."""
    maxima = local_maxima(prices)
    if len(maxima) < 3:
        return 0
    left, mid, right = maxima[-3], maxima[-2], maxima[-1]
    if prices[mid] > prices[left] and prices[mid] > prices[right]:
        if abs(prices[left] - prices[right]) / max(prices[left], prices[right]) < tolerance:
            if (prices[mid] - max(prices[left], prices[right])) / max(prices[left], prices[right]) > tolerance:
                return 1
    return 0

def detect_inverse_head_and_shoulders(prices, tolerance=0.03):
    """Detect a simple inverse head and shoulders pattern."""
    minima = local_minima(prices)
    if len(minima) < 3:
        return 0
    left, mid, right = minima[-3], minima[-2], minima[-1]
    if prices[mid] < prices[left] and prices[mid] < prices[right]:
        if abs(prices[left] - prices[right]) / max(prices[left], prices[right]) < tolerance:
            if (min(prices[left], prices[right]) - prices[mid]) / min(prices[left], prices[right]) > tolerance:
                return 1
    return 0

def detect_double_top(prices, tolerance=0.02):
    """Detect a double top pattern."""
    maxima = local_maxima(prices)
    if len(maxima) < 2:
        return 0
    if abs(prices[maxima[-2]] - prices[maxima[-1]]) / max(prices[maxima[-2]], prices[maxima[-1]]) < tolerance:
        return 1
    return 0

def detect_double_bottom(prices, tolerance=0.02):
    """Detect a double bottom pattern."""
    minima = local_minima(prices)
    if len(minima) < 2:
        return 0
    if abs(prices[minima[-2]] - prices[minima[-1]]) / max(prices[minima[-2]], prices[minima[-1]]) < tolerance:
        return 1
    return 0

def detect_breakout(prices, window=20, threshold=0.02):
    """
    Detect a breakout if the current price is higher than the max or lower than the min 
    of the recent window by a given threshold.
    """
    if len(prices) < window:
        return 0
    recent = prices[-window:]
    current = prices[-1]
    if current > np.max(recent) * (1 + threshold) or current < np.min(recent) * (1 - threshold):
        return 1
    return 0

# -------------------------------
# 3. Additional helper for computing ATR
# -------------------------------
def compute_atr(data, period=14):
    """Compute the Average True Range (ATR) over the given period."""
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# -------------------------------
# 4. Data fetching and enhanced feature engineering
# -------------------------------
def fetch_data(symbol, start_date, end_date):
    """Fetch historical OHLCV data for the given symbol."""
    data = yf.download(symbol, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def feature_engineering(data, symbol):
    """
    Compute technical indicators, chart pattern flags, and additional macro/alternative features.
    This version removes sentiment and adds:
      - ROC_10: 10-day rate of change.
      - Momentum_10: Difference between the current close and the close 10 days ago.
      - ATR_14: 14-day Average True Range.
    """
    # Basic technical indicators
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=10).std()
    
    # RSI (14-day)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD and MACD Signal
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands (20-day)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['STD_20'] = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['SMA_20'] + (2 * data['STD_20'])
    data['Bollinger_Lower'] = data['SMA_20'] - (2 * data['STD_20'])
    
    # OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i].item() > data['Close'].iloc[i-1].item():
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i].item() < data['Close'].iloc[i-1].item():
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    
    # Additional momentum and regime features:
    data['ROC_10'] = data['Close'].pct_change(periods=10)  # 10-day rate of change (percentage)
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)  # Absolute momentum
    data['ATR_14'] = compute_atr(data, period=14)  # 14-day Average True Range
    
    # Chart pattern detection flags (20-day rolling window)
    pattern_window = 20
    head_shoulder = [0] * pattern_window
    inverse_head_shoulder = [0] * pattern_window
    double_top = [0] * pattern_window
    double_bottom = [0] * pattern_window
    breakout = [0] * pattern_window

    for i in range(pattern_window, len(data)):
        window_prices = data['Close'].iloc[i - pattern_window:i].values
        head_shoulder.append(detect_head_and_shoulders(window_prices))
        inverse_head_shoulder.append(detect_inverse_head_and_shoulders(window_prices))
        double_top.append(detect_double_top(window_prices))
        double_bottom.append(detect_double_bottom(window_prices))
        breakout.append(detect_breakout(window_prices, window=pattern_window, threshold=0.02))
    
    data['HeadShoulders'] = head_shoulder
    data['InverseHeadShoulders'] = inverse_head_shoulder
    data['DoubleTop'] = double_top
    data['DoubleBottom'] = double_bottom
    data['Breakout'] = breakout

    data.dropna(inplace=True)
    return data

# -------------------------------
# 5. Create sequences for the model
# -------------------------------
def create_sequences(data, window_size, target_col='Close'):
    """
    Create overlapping sequences for model input.
    Each sample consists of a sequence of days (with multiple features),
    and the target is the closing price immediately after the sequence.
    """
    # Feature columns including technical indicators, chart patterns, momentum, macro data, and index returns.
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_50', 'Return', 'Volatility', 'RSI',
        'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'OBV',
        'HeadShoulders', 'InverseHeadShoulders', 'DoubleTop', 'Doublottom', 'Breakout',
        'ROC_10', 'ATR_14', 'Momentum_10', 'VIX', 'TNX'
    ]
    data_features = data[feature_cols].values
    data_target = data[target_col].values

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data_features[i - window_size:i])
        y.append(data_target[i])
    return np.array(X), np.array(y)

# -------------------------------
# 6. Build the improved model with attention (Hybrid CNN-LSTM)
# -------------------------------
def build_improved_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = AttentionLayer()(x)
    x = Dense(25, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()
    return model

# -------------------------------
# 7. Train the model
# -------------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)
    return history

# -------------------------------
# 8. Main pipeline: Data preparation, training, and prediction for the next 50 days
# -------------------------------
if __name__ == "__main__":
    # Parameters
    symbol = "AAPL"
    start_date = "2017-01-01"
    end_date = "2025-01-01"
    window_size = 30

    # 1. Fetch main asset data
    print("Fetching main asset data...")
    data = fetch_data(symbol, start_date, end_date)
    
    # 3. Fetch additional macro data: VIX and 10-year Treasury yield (TNX)
    print("Fetching VIX and TNX data...")
    vix = yf.download("^VIX", start=start_date, end=end_date)
    tnx = yf.download("^TNX", start=start_date, end=end_date)
    # Rename the 'Close' column to identify the feature
    vix = vix[['Close']].rename(columns={"Close": "VIX"})
    tnx = tnx[['Close']].rename(columns={"Close": "TNX"})
    
    # Merge Nasdaq, S&P 500, VIX, and TNX data into the main data (join on date index)
    data = data.join(vix, how='left')
    data = data.join(tnx, how='left')
    data['VIX'].fillna(method='ffill', inplace=True)
    data['TNX'].fillna(method='ffill', inplace=True)
    
    # 4. Perform enhanced feature engineering (macro/alternative data and improved momentum measures)
    print("Performing feature engineering...")
    data = feature_engineering(data, symbol)
    
    # Create a date series corresponding to sequences (starting at index "window_size")
    dates = data.index[window_size:]
    
    # 5. Create sequences from the data
    print("Creating sequences...")
    X, y = create_sequences(data, window_size, target_col='Close')
    
    # 6. Sequentially split the data into training (80%) and validation (20%)
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    dates_train = dates[:split_index]
    dates_val = dates[split_index:]
    
    # 7. Normalize features
    num_features = X.shape[2]
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    
    # 8. Build and train the improved model with attention
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    improved_model = build_improved_model(input_shape)
    print("Training the improved model with macro and momentum features...")
    train_model(improved_model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=100, batch_size=32)
    
    # 9. Predict the price for the next 50 days and compare with actual prices
    print("Evaluating predictions for the next 50 days...")
    if len(X_val_scaled) >= 50:
        X_test = X_val_scaled[-50:]
        y_test = y_val[-50:]
        test_dates = dates_val[-50:]
    else:
        X_test = X_train_scaled[-50:]
        y_test = y_train[-50:]
        test_dates = dates_train[-50:]
    
    test_dates = [d.strftime("%Y-%m-%d") for d in list(test_dates)]
    
    predicted_prices = improved_model.predict(X_test)
    y_test = np.array(y_test).flatten()
    predicted_prices = np.array(predicted_prices).flatten()
    
    results = pd.DataFrame({
        "Date": test_dates,
        "Actual Price": y_test,
        "Predicted Price": predicted_prices,
        "Error Difference": y_test - predicted_prices
    })
    
    print("\n50-Day Prediction Results with Macro Data and Enhanced Momentum Features:")
    print(results.to_string(index=False))