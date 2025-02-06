import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
from statsmodels.tsa.holtwinters import Holt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Load the dataset
data = pd.read_csv('GoldUP.csv')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Format the Date column to 'yyyy' format (only the year)
data['Date'] = data['Date'].dt.year

# Select relevant columns
data = data[['Date', 'Gold_Price', 'Crude_Oil', 'Interest_Rate', 'USD_INR', 'Sensex', 'CPI', 'USD_Index']]

# -----------------------------------------------------------------------------
# Model 1: Multiple Linear Regression
# -----------------------------------------------------------------------------
# Define features (independent variables) and target (dependent variable)
X = data[['Crude_Oil', 'Interest_Rate', 'USD_INR', 'Sensex', 'CPI', 'USD_Index']]
y = data['Gold_Price']

# Fit Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X, y)

# Predict using the model
data['LR_Forecast'] = model_lr.predict(X)

# Plotting the chart for Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Gold_Price'], label='Actual Gold Price', color='black')
plt.plot(data['Date'], data['LR_Forecast'], label='Linear Regression Forecast', color='blue')
plt.title('Gold Price Forecasting - Multiple Linear Regression')
plt.xlabel('Year')
plt.ylabel('Gold Price (INR)')
plt.xticks(rotation=45)
plt.xticks(np.arange(min(data['Date']), max(data['Date']) + 1, 1))
plt.legend()
plt.show()

# Calculate MSE, MAPE, RMSE for Linear Regression
mse_lr = mean_squared_error(data['Gold_Price'], data['LR_Forecast'])
mape_lr = mean_absolute_percentage_error(data['Gold_Price'], data['LR_Forecast'])
rmse_lr = sqrt(mse_lr)

print(f"Linear Regression - MSE: {mse_lr:.2f}, MAPE: {mape_lr:.2f}, RMSE: {rmse_lr:.2f}")

# -----------------------------------------------------------------------------
# Model 2: Moving Average (6 months)
# -----------------------------------------------------------------------------
# Calculate Moving Average (6 months)
data['MA_6'] = data['Gold_Price'].rolling(window=6).mean()

# Plotting the chart for Moving Average (6 months)
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Gold_Price'], label='Actual Gold Price', color='black')
plt.plot(data['Date'], data['MA_6'], label='6-Month Moving Average', color='blue')
plt.title('Gold Price Forecasting - 6 Month Moving Average')
plt.xlabel('Year')
plt.ylabel('Gold Price (INR)')
plt.xticks(rotation=45)
plt.xticks(np.arange(min(data['Date']), max(data['Date']) + 1, 1))
plt.legend()
plt.show()

# Calculate MSE, MAPE, RMSE for MA(6)
mse_6 = mean_squared_error(data['Gold_Price'][5:], data['MA_6'][5:])
mape_6 = mean_absolute_percentage_error(data['Gold_Price'][5:], data['MA_6'][5:])
rmse_6 = sqrt(mse_6)

print(f"Moving Average (6 months) - MSE: {mse_6:.2f}, MAPE: {mape_6:.2f}, RMSE: {rmse_6:.2f}")

# -----------------------------------------------------------------------------
# Model 3: Moving Average (12 months)
# -----------------------------------------------------------------------------
# Calculate Moving Average (12 months)
data['MA_12'] = data['Gold_Price'].rolling(window=12).mean()

# Plotting the chart for Moving Average (12 months)
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Gold_Price'], label='Actual Gold Price', color='black')
plt.plot(data['Date'], data['MA_12'], label='12-Month Moving Average', color='blue')
plt.title('Gold Price Forecasting - 12 Month Moving Average')
plt.xlabel('Year')
plt.ylabel('Gold Price (INR)')
plt.xticks(rotation=45)
plt.xticks(np.arange(min(data['Date']), max(data['Date']) + 1, 1))
plt.legend()
plt.show()

# Calculate MSE, MAPE, RMSE for MA(12)
mse_12 = mean_squared_error(data['Gold_Price'][11:], data['MA_12'][11:])
mape_12 = mean_absolute_percentage_error(data['Gold_Price'][11:], data['MA_12'][11:])
rmse_12 = sqrt(mse_12)

print(f"Moving Average (12 months) - MSE: {mse_12:.2f}, MAPE: {mape_12:.2f}, RMSE: {rmse_12:.2f}")

# -----------------------------------------------------------------------------
# Model 4: Simple Exponential Smoothing
# -----------------------------------------------------------------------------
# Fit Simple Exponential Smoothing model
model_se = SimpleExpSmoothing(data['Gold_Price'])
model_se_fit = model_se.fit(smoothing_level=0.2, optimized=False)

# Forecast
data['SES_Forecast'] = model_se_fit.fittedvalues

# Plotting the chart for Simple Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Gold_Price'], label='Actual Gold Price', color='black')
plt.plot(data['Date'], data['SES_Forecast'], label='Simple Exponential Smoothing Forecast', color='blue')
plt.title('Gold Price Forecasting - Simple Exponential Smoothing')
plt.xlabel('Year')
plt.ylabel('Gold Price (INR)')
plt.xticks(rotation=45)
plt.xticks(np.arange(min(data['Date']), max(data['Date']) + 1, 1))
plt.legend()
plt.show()

# Calculate MSE, MAPE, RMSE for SES
mse_ses = mean_squared_error(data['Gold_Price'], data['SES_Forecast'])
mape_ses = mean_absolute_percentage_error(data['Gold_Price'], data['SES_Forecast'])
rmse_ses = sqrt(mse_ses)

print(f"Simple Exponential Smoothing - MSE: {mse_ses:.2f}, MAPE: {mape_ses:.2f}, RMSE: {rmse_ses:.2f}")

# -----------------------------------------------------------------------------
# Model 5: Holt's Method
# -----------------------------------------------------------------------------
# Fit Holt's model (Linear trend)
model_holt = Holt(data['Gold_Price'])
model_holt_fit = model_holt.fit(smoothing_level=0.8, smoothing_trend=0.2)

# Forecast
data['Holt_Forecast'] = model_holt_fit.fittedvalues

# Plotting the chart for Holt's method
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Gold_Price'], label='Actual Gold Price', color='black')
plt.plot(data['Date'], data['Holt_Forecast'], label='Holt\'s Method Forecast', color='blue')
plt.title('Gold Price Forecasting - Holt\'s Method')
plt.xlabel('Year')
plt.ylabel('Gold Price (INR)')
plt.xticks(rotation=45)
plt.xticks(np.arange(min(data['Date']), max(data['Date']) + 1, 1))
plt.legend()
plt.show()

# Calculate MSE, MAPE, RMSE for Holt's method
mse_holt = mean_squared_error(data['Gold_Price'], data['Holt_Forecast'])
mape_holt = mean_absolute_percentage_error(data['Gold_Price'], data['Holt_Forecast'])
rmse_holt = sqrt(mse_holt)

print(f"Holt's Method - MSE: {mse_holt:.2f}, MAPE: {mape_holt:.2f}, RMSE: {rmse_holt:.2f}")

# -----------------------------------------------------------------------------
# Model Comparison
# -----------------------------------------------------------------------------
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Moving Average (6 Months)', 'Moving Average (12 Months)', 'Simple Exponential Smoothing', 'Holt\'s Method'],
    'MSE': [mse_lr, mse_6, mse_12, mse_ses, mse_holt],
    'MAPE': [mape_lr, mape_6, mape_12, mape_ses, mape_holt],
    'RMSE': [rmse_lr, rmse_6, rmse_12, rmse_ses, rmse_holt]
})

# Display the comparison table
print(comparison_df)

# Inference based on MAPE
best_model = comparison_df.loc[comparison_df['MAPE'].idxmin(), 'Model']
print(f"The best model based on MAPE is: {best_model}")

