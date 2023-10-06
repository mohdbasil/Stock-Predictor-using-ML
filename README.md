# Stock-Predictor-using-ML
This GitHub repository contains the source code and materials for a Python project focused on predicting stock prices using the Exponential Moving Average (EMA) technique. The project's goal is to provide accurate stock price forecasts and evaluate prediction model performance
Stock Data : https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1665056459&period2=1696592459&interval=1d&events=history&includeAdjustedClose=true
src/ :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load historical stock price data (you can replace this with your data source)
# Ensure your data has columns 'Date' and 'Close' for the date and closing prices.
data = pd.read_csv("~/Downloads/AAPL.csv")

# Calculate the EMA
def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Choose an EMA window (you can tune this parameter)
ema_window = 10

# Calculate EMA
data['EMA'] = calculate_ema(data, ema_window)

# Visualize the stock price and EMA
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Stock Price', color='blue')
plt.plot(data['Date'], data['EMA'], label=f'EMA ({ema_window} periods)', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price vs. Exponential Moving Average')
plt.legend()
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Predict stock prices using EMA
def predict_ema(data, window):
    predictions = []
    for i in range(len(data)):
        if i < window:
            predictions.append(None)
        else:
            ema_values = data['EMA'].values[i-window:i]
            ema_mean = np.mean(ema_values)
            predictions.append(ema_mean)
    return predictions

# Predict using EMA
test_data['EMA_Predictions'] = predict_ema(test_data, ema_window)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test_data['Close'][ema_window:], test_data['EMA_Predictions'][ema_window:])
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_data['Close'][ema_window:], test_data['EMA_Predictions'][ema_window:])
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((test_data['Close'][ema_window:] - test_data['EMA_Predictions'][ema_window:]) / test_data['Close'][ema_window:]) * 100)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Visualize the predictions
plt.figure(figsize=(12,6))
plt.plot(test_data['Date'], test_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(test_data['Date'], test_data['EMA_Predictions'], label=f'Predicted EMA ({ema_window} periods)', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual Stock Price vs. Predicted EMA')
plt.legend()
plt.grid(True)
plt.show()
