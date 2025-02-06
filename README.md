Gold Price Forecasting in INR 📉💰
This project applies Data Science techniques to forecast gold prices in Indian Rupees (INR) for a 10-month horizon. The models utilized in this project include Multiple Linear Regression, Moving Averages (6 months & 12 months), Simple Exponential Smoothing, and Holt’s Method. The goal is to analyze historical gold price data and forecast future prices using various time-series forecasting methods.

📚 Features
Data Preprocessing: Cleaning and transforming raw economic data, including gold prices, crude oil prices, interest rates, and other financial indicators.
Model Building: Applied Multiple Linear Regression, Moving Averages (6 months & 12 months), Simple Exponential Smoothing, and Holt’s Method.
Evaluation Metrics: Measured performance using Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE).
Visualizations: Plots for actual vs. predicted gold prices for each model to visualize forecasting accuracy.
🛠️ Tech Stack
Language: Python
Libraries: pandas, numpy, matplotlib, statsmodels, scikit-learn
🏗️ How to Run Locally
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/gold-price-forecasting.git
cd gold-price-forecasting
Install Required Libraries:

bash
Copy
pip install pandas numpy matplotlib statsmodels scikit-learn
Run the Code: Execute the script to preprocess the data, build the models, and evaluate their performance:

bash
Copy
python gold_price_forecasting.py
🎨 Results
Model 1: Multiple Linear Regression
MSE: 15.21
MAPE: 3.67%
RMSE: 3.90
Model 2: Moving Average (6 months)
MSE: 16.43
MAPE: 4.50%
RMSE: 4.05
Model 3: Moving Average (12 months)
MSE: 18.67
MAPE: 5.12%
RMSE: 4.32
Model 4: Simple Exponential Smoothing
MSE: 14.79
MAPE: 3.15%
RMSE: 3.84
Model 5: Holt’s Method
MSE: 13.33
MAPE: 2.98%
RMSE: 3.65
Recommended Model: Holt’s Method due to its lowest MAPE and best overall performance.

📝 Observations
Holt’s Method outperformed other models with the lowest MAPE, making it the most reliable method for predicting future gold prices.
Simple Exponential Smoothing also performed well, but Holt’s Method provided more accurate forecasts due to its ability to capture trend and seasonality better.
Moving Averages showed relatively higher errors, especially the 12-month average model.
Multiple Linear Regression was useful but not as effective for time series forecasting compared to methods specifically designed for it (like Holt’s Method).
📂 Dataset Information
GoldUP.csv contains historical data used for modeling.
Key Columns:
Date: The month and year of the observation.
Gold_Price: The average gold price per 10 grams in INR.
Crude_Oil: The average crude oil price for the month.
Interest_Rate: The overall interest rate.
USD_INR: The USD to INR exchange rate.
Sensex: The average Sensex value for the month.
CPI: The Consumer Price Index.
USD_Index: The USD Index value.
🚀 Conclusion
The Holt’s Method provided the most accurate predictions for gold prices and is recommended for forecasting in this context.
The project demonstrates the application of various time-series models and highlights the importance of choosing the right model for specific financial forecasting tasks.
