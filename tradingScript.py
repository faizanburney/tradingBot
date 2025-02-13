import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

###########################
# 1. Data Fetching Function
###########################
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.sort_index(inplace=True)
    data.dropna(inplace=True)
    return data

###########################
# 2. Feature Engineering Function
###########################
def feature_engineering(data):
    # Compute technical indicators
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
    

    # Additional Macro/Alternative Data (simulated)
    np.random.seed(42)
    data['Interest_Rate'] = np.random.uniform(0.01, 0.05, size=len(data))  # simulated interest rates (1%-5%)
    data['VIX'] = np.random.uniform(15, 35, size=len(data))  # simulated VIX values

    data.dropna(inplace=True)
    return data

###########################
# 3. Sequence Creation Function
###########################
def create_sequences(data, window_size, target_col='Close'):
    # Define feature columns for the model
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_10', 'SMA_50', 'Return', 'Volatility', 'RSI',
                    'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower',
                    'Interest_Rate', 'VIX']
    data_features = data[feature_cols].values
    data_target = data[target_col].values
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data_features[i-window_size:i])
        y.append(data_target[i])
    return np.array(X), np.array(y)

###########################
# 4. LSTM-Only Model Function (with target scaling)
###########################
def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, recurrent_dropout_rate=0.2, num_layers=2):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False
        x = LSTM(lstm_units, return_sequences=return_seq, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()
    return model

###########################
# 5. Model Training Function
###########################
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)
    return history

###########################
# 6. Main Pipeline: Data Prep, Training, Prediction, and Plotting
###########################
if __name__ == "__main__":
    # Parameters
    symbol = "AAPL"
    start_date = "2018-01-01"
    end_date = "2024-01-01"
    window_size = 60

    # Fetch and prepare data
    print("Fetching main asset data...")
    data = fetch_data(symbol, start_date, end_date)
    
    print("Performing feature engineering...")
    data = feature_engineering(data)
    
    # Create a date series for sequences (starting at index 'window_size')
    dates = data.index[window_size:]
    
    print("Creating sequences...")
    X, y = create_sequences(data, window_size, target_col='Close')
    
    # Sequentially split data: 80% training, 20% validation
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    dates_train, dates_val = dates[:split_index], dates[split_index:]
    
    # Normalize input features
    num_features = X.shape[2]
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)
    scaler_X = StandardScaler()
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    
    # Scale target variable (Y) for LSTM training
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
    y_val_scaled = scaler_y.transform(np.array(y_val).reshape(-1, 1))
    
    # Build and train the LSTM model on scaled Y
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    lstm_model = build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, recurrent_dropout_rate=0.2, num_layers=2)
    print("Training the LSTM model (with target scaling)...")
    train_model(lstm_model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, epochs=50, batch_size=32)
    
    # Prepare test set: use the last 50 samples from validation if available
    if len(X_val_scaled) >= 50:
        X_test = X_val_scaled[-50:]
        y_test = y_val[-50:]  # in original scale
        test_dates = dates_val[-50:]
    else:
        X_test = X_train_scaled[-50:]
        y_test = y_train[-50:]
        test_dates = dates_train[-50:]
    
    # Convert test_dates to list of strings
    test_dates = [d.strftime("%Y-%m-%d") for d in list(test_dates)]

    # Make predictions using the LSTM model (predictions are in scaled space)
    lstm_pred_scaled = lstm_model.predict(X_test)
    # Invert scaling to get predictions in original price scale
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()

    y_test = np.array(y_test).flatten()

    # Create a results DataFrame
    results = pd.DataFrame({
        "Date": test_dates,
        "Actual Price": y_test,
        "LSTM Prediction": lstm_pred,
        "Error": y_test - lstm_pred
    })

    print("\n50-Day Prediction Results (LSTM Only):")
    print(results.to_string(index=False))

    ###########################
    # Plotting: Line and Scatter Plots
    ###########################
    # Ensure 'Date' column is in datetime format for plotting
    results['Date'] = pd.to_datetime(results['Date'])

    # Line Plot: Actual vs. LSTM Predicted Price over time
    plt.figure(figsize=(14, 7))
    plt.plot(results['Date'], results['Actual Price'], label="Actual Price", marker='o')
    plt.plot(results['Date'], results['LSTM Prediction'], label="LSTM Prediction", marker='x')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual Price vs. LSTM Predicted Price (Line Plot)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Scatter Plot: Actual Price vs. LSTM Predicted Price
    plt.figure(figsize=(8, 6))
    plt.scatter(results['Actual Price'], results['LSTM Prediction'], color='tab:blue', alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("LSTM Predicted Price")
    plt.title("Scatter Plot: Actual vs. LSTM Predicted Price")
    # Plot 45-degree line for reference
    min_price = results['Actual Price'].min()
    max_price = results['Actual Price'].max()
    plt.plot([min_price, max_price], [min_price, max_price], color='red', linestyle='--')
    plt.tight_layout()
    plt.show()