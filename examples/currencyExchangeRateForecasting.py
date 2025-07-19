#%% md
# # **Forecasting Currency Exchange Rates Using OpenBB Historical Data**
# This notebook demonstrates how to predict future movements in currency exchange rates
# using OpenBB's historical data. This notebook builds different forecasting model capable
# of analyzing trends in currency pairs such as USD/EUR, enabling data-driven predictions
# for future rates. The models evaluates risk and potential returns, providing valuable insights
# for traders, investors, and financial analysts.
# 
# [![Author Profile]()](https://github.com/Manish-k723)
# 
# 
# !pip install openbb -q #uncommment if you are in google colab
# !pip install pmdarima -q

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

from openbb import obb  # Fetches historical forex data from OpenBB
import pandas as pd  # Data manipulation and analysis

import numpy as np  # For numerical computations
print(np.__version__) # MUSS 1.26.4 sein!!!
# pip install numpy==1.26.4
# pip install --upgrade numpy

import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Evaluation of model performance (e.g., MSE)
from sklearn.preprocessing import MinMaxScaler  # Data normalization (scaling values)
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Seasonal ARIMA forecasting model
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Exponential smoothing for time-series
import pmdarima as pm  # Auto-ARIMA for automatic ARIMA parameter selection
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout  # LSTM neural network layers for time-series data



#%% md
# # **Loading Data**
# This cell fetches historical exchange rate data for the EUR/USD pair using the yfinance provider, choose the provider accordingly.
# 
# Please Refer [Yfinance](https://pypi.org/project/yfinance/) documenation for list of currency exchange symbols.
#%%
# Fetching historical data for the EUR/USD pair using the yfinance provider
start_date = '1991-01-01'
end_date = '2025-07-18'

# Since yfinance uses "EURUSD=X", we'll use that
forex_df = obb.equity.price.historical(symbol="EURUSD=X", provider="yfinance", start_date=start_date, end_date=end_date).to_df()

forex_df.sample(5)
#%% md
# # **Data Preprocessing**
#%%
forex_df.index = pd.to_datetime(forex_df.index)

forex_df = forex_df.asfreq('D')  # Resamples the data to a daily frequency ('D' stands for days), ensuring data is indexed daily

forex_df.ffill(inplace=True) # Forward fills missing values to fill gaps in the time series with the last available value
#%%
print(forex_df.info())
#%%
# Split the data, keeping 20% of it for testing
train_size = int(len(forex_df) * 0.8)
train_data, test_data = forex_df['close'][:train_size], forex_df['close'][train_size:]

print(f"Training data: {len(train_data)} rows")
print(f"Testing data: {len(test_data)} rows")

# # **Model Training & Prediction**
# 
# In this section, we focus on time-series forecasting, which differs from traditional
# machine learning tasks. Unlike predicting a single output variable, time-series models aim to predict
# future values based on historical data, considering the sequential nature of the data. This is
# particularly important for predicting currency exchange rates, where trends, seasonality, and past
# values heavily influence future movements.
#%% md
# # **ARIMA Model**
# 
# We will start by using the ARIMA (AutoRegressive Integrated Moving Average) model for time-series forecasting. ARIMA is one of the most popular models for time-series analysis, as it combines three components:
# 
# 1. AR (AutoRegressive): Uses past values to predict future ones.
# 2. I (Integrated): Makes the series stationary by differencing it.
# 3. MA (Moving Average): Models the error terms from previous time steps.
# 
# This model is ideal for capturing the trends and patterns in the currency exchange data. Let’s train and evaluate it on our dataset.
#%%
# %%time
auto_arima_model = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True, ensure_all_finite=True)
arima_predictions = auto_arima_model.predict(n_periods=len(test_data))
print(auto_arima_model.summary())
#%% md
# # **SARIMAX Model**
# Next, we will explore the SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) model for forecasting. SARIMAX is an extension of the ARIMA model that incorporates seasonality and exogenous variables (optional external factors) into the prediction process.
# 
# 1. Seasonality: Captures repeating patterns over a fixed period (e.g., weekly or monthly cycles).
# 2. Exogenous Variables (X): Allows the model to include additional factors that may influence the target variable (optional).
# 
# SARIMAX is particularly useful when dealing with time-series data that exhibits periodic fluctuations, making it well-suited for forecasting currency exchange rates where trends may repeat over time. Let’s apply SARIMAX to our dataset.
#%%
# %%time
sarimax_model = SARIMAX(train_data,
                       order=(5, 1, 0),  # non-seasonal order
                       seasonal_order=(1, 1, 1, 12),  # seasonal order: parameters tuning is required
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarimax_fit = sarimax_model.fit(disp=False)
sarimax_predictions = sarimax_fit.forecast(steps=len(test_data))
#%% md
# # **Exponential Smoothing**
# We will also use the Exponential Smoothing technique for time-series forecasting. Unlike ARIMA and SARIMAX, this method places greater emphasis on more recent observations, making it useful for capturing short-term trends. Exponential smoothing can model various components of time series data, such as:
# 
# 1. Level: The baseline value of the series.
# 2. Trend: The overall direction of the series.
# 3. Seasonality: The repeating short-term patterns.
# 
# This method is particularly effective for forecasting time-series data with trends and seasonality, making it suitable for currency exchange rate prediction, where both short- and long-term movements need to be captured.
#%%

exp_smooth_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)
exp_smooth_fit = exp_smooth_model.fit()

# Predict using Exponential Smoothing
exp_smooth_predictions = exp_smooth_fit.forecast(steps=len(test_data))
#%% md
# # **LSTM Model**
# Finally, we will employ a Long Short-Term Memory (LSTM) model, a type of recurrent neural network (RNN) specifically designed to handle sequential data like time series. LSTMs excel at capturing long-term dependencies in data by using memory cells that can retain information over extended time periods, which makes them well-suited for tasks where past values influence future ones, such as currency exchange rate prediction.
# 
# LSTMs are particularly powerful for modeling complex, non-linear relationships in time series data, making them ideal for forecasting in dynamic environments like financial markets, where historical patterns may vary in unexpected ways.
#%%

# Step 1: Data Preparation

# Scale the close prices of train_data and test_data (Series)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))

# Creating dataset for LSTM from train_data
X_train, y_train = [], []
for i in range(60, len(scaled_train_data)):
    X_train.append(scaled_train_data[i-60:i, 0])  # Previous 60 days
    y_train.append(scaled_train_data[i, 0])  # Current day
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Step 2: Build and Compile LSTM Model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # Fix: Use Input layer
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Train the Model on the training data
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

# Step 4: Preparing the test_data for making predictions

# Creating the test data sequences (just like we did for train_data)
X_test = []
for i in range(60, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-60:i, 0])  # Previous 60 days
X_test = np.array(X_test)

# Reshaping for LSTM
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 5: Make Predictions on test_data
lstm_predictions = model.predict(X_test)

# Inverse scaling to get actual values for predictions
lstm_predictions = scaler.inverse_transform(lstm_predictions)
#%%
comparison_df = test_data.reset_index()
comparison_df['arima_predictions'] = arima_predictions.reset_index(drop=True)
comparison_df['sarimax_predictions'] = sarimax_predictions.reset_index(drop=True)
comparison_df['exp_smooth_predictions'] = exp_smooth_predictions.reset_index(drop=True)
comparison_df['lstm_predictions'] = np.nan
comparison_df.loc[60:, 'lstm_predictions'] = lstm_predictions.flatten()
comparison_df.sample(10)
#%%
# Plotting the actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(comparison_df['date'], comparison_df['close'], label='Actual Close Price', color='blue')
plt.plot(comparison_df['date'], comparison_df['arima_predictions'], label='ARIMA Predictions', color='orange')
plt.plot(comparison_df['date'], comparison_df['sarimax_predictions'], label='SARIMAX Predictions', color='green')
plt.plot(comparison_df['date'], comparison_df['exp_smooth_predictions'], label='Exponential Smoothing Predictions', color='red')
plt.plot(comparison_df['date'], comparison_df['lstm_predictions'], label='Long Short Term Memomy Predictions', color='violet')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
#%%
# Calculate errors for each prediction method
metrics = {}

# Define a function to calculate metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, mse, rmse, mape

# Get actual values
actual_values = comparison_df['close'].values

# Dropping rows where any of the predictions are NaN for cleaning
comparison_df_clean = comparison_df.dropna(subset=['arima_predictions', 'sarimax_predictions', 'exp_smooth_predictions', 'lstm_predictions'])

# Get the cleaned actual and predicted values
actual_values_clean = comparison_df_clean['close'].values
arima_predictions_clean = comparison_df_clean['arima_predictions'].values
sarimax_predictions_clean = comparison_df_clean['sarimax_predictions'].values
exp_smooth_predictions_clean = comparison_df_clean['exp_smooth_predictions'].values
lstm_predictions_clean = comparison_df_clean['lstm_predictions'].values

# Calculate metrics for each prediction method
metrics['ARIMA'] = calculate_metrics(actual_values_clean, arima_predictions_clean)
metrics['SARIMAX'] = calculate_metrics(actual_values_clean, sarimax_predictions_clean)
metrics['Exp.Smoothing'] = calculate_metrics(actual_values_clean, exp_smooth_predictions_clean)
metrics['LSTM'] = calculate_metrics(actual_values_clean, lstm_predictions_clean)

# Create a summary DataFrame
metrics_df = pd.DataFrame(metrics, index=['MAE', 'MSE', 'RMSE', 'MAPE']).T
metrics_df.columns = ['MeanAbsErr', 'MeanSquErr', 'RootMeanSquErr', 'MeanAbsPercErr']
# ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error']
plt.figure(figsize=(10, 6))

# Create a heatmap
sns.heatmap(metrics_df, annot=True, fmt='.6f', linewidths=0.1, vmax=1.0, vmin=-1.0, cbar=True, cmap=plt.cm.RdBu_r, linecolor='white')

# Adding titles and labels
plt.title('Model Performance Comparison')
plt.xlabel('Metrics')
plt.ylabel('Models')
# Show the plot
plt.show()

#%% md
# # **Hyperparameter Tuning of SARIMAX**
# 
# In this section, we conduct hyperparameter tuning for the SARIMAX model to find the optimal combination of parameters that minimizes the Akaike Information Criterion (AIC). The AIC is a measure of the goodness of fit of a statistical model, and lower values indicate a better fit.
# 
# 
# ##### **Note: Hyperparameter tuning step consumes a lot of time(more than a hour),
# Beneath provided is just sample code for usage if you have enough resourse and time then only try it after uncommenting.**
# 
# 
# 
#%%
# # Define the p, d, q parameters to take any value between 0 and 2
#import itertools
# p = d = q = range(0, 3)

# # Define the seasonal parameters (P, D, Q, s)
# P = D = Q = range(0, 2)
# seasonal_period = [7, 14, 21]  # Seasonal period, e.g., 12 for monthly data

# # Create a list of all possible combinations of p, d, q for non-seasonal and seasonal terms
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = list(itertools.product(P, D, Q, seasonal_period))

# # Search for the best combination of parameters
# best_aic = np.inf
# best_pdq = None
# best_seasonal_pdq = None
# best_model = None

# for param in pdq:
#     for seasonal_param in seasonal_pdq:
#         try:
#             # Fit the SARIMAX model with the given parameters
#             model = SARIMAX(train_data,
#                             order=param,
#                             seasonal_order=seasonal_param,
#                             enforce_stationarity=False,
#                             enforce_invertibility=False)
#             results = model.fit(disp=False)

#             # Keep track of the best model based on AIC
#             if results.aic < best_aic:
#                 best_aic = results.aic
#                 best_pdq = param
#                 best_seasonal_pdq = seasonal_param
#                 best_model = results

#         except Exception as e:
#             continue

# print(f"Best SARIMAX model: ARIMA{best_pdq} x {best_seasonal_pdq}12 - AIC: {best_aic}")
#%%
# sarima_predictions = best_model.forecast(steps=len(test_data))

# # Compute the RMSE
# rmse = np.sqrt(mean_squared_error(test_data, sarima_predictions))
# print(f"SARIMAX Test RMSE: {rmse}")