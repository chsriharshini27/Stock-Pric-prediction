# Stock-Market-prediction


DATA COLLECTION:

Sources: Stock price data can be collected from financial APIs like Alpha Vantage, Yahoo Finance, or Quandl.
Features: Common features include historical prices, trading volumes, technical indicators (e.g., moving averages, RSI), and macroeconomic indicators (e.g., interest rates, GDP).

import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

data = download_stock_data('AAPL', '2020-01-01', '2023-01-01')
data.to_csv('data/raw/aapl.csv')


DATA PROCESSING :

Cleaning: Handle missing values, outliers, and ensure data consistency.
Normalization/Standardization: Scale features to a standard range to improve model performance.
Feature Engineering: Create additional features that could help in prediction, like moving averages, price momentum, etc.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.fillna(method='ffill', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data

scaled_data = preprocess_data('data/raw/aapl.csv')


MODEL TRAINING:

Train-Test Split: Split data into training and testing sets.
Cross-Validation: Use techniques like k-fold cross-validation to ensure model robustness.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X, y = create_dataset(scaled_data, time_step)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)


EVALUATION :

Metrics: Common metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared, etc.
Visualization: Plot predictions vs. actual values to visually inspect model performance.

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(predictions, actual):
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    print(f'MSE: {mse}, MAE: {mae}')

evaluate_model(predictions, y_test)

