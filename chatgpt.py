import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_log_error

# Customize the app theme
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",
    )

# Define the app title and description
st.title("Sales Prediction App")
st.write("""
         This app predicts sales using an autoregression model. 
         Use the sliders to set the forecast period and lag order.
         """)

# Load the sales data
sales_data = pd.read_csv("sales_data.csv")

# Define the user input form
st.sidebar.title("User Input")
forecast_period = st.sidebar.slider("Select forecast period (in days)", 1, 365, 30)
lag_order = st.sidebar.slider("Select lag order (p)", 1, 20, 3)

# Train the autoregression model
model = AutoReg(sales_data, lags=lag_order)
result = model.fit()

# Calculate RMSLE score for each lag order
x = sales_data
X_train, X_test = x[:1673], x[1673:]
armodel = AutoReg(X_train, lags=lag_order).fit()
pred = armodel.predict(start=len(X_train), end=len(x)-1, dynamic=False)
rmsle = np.sqrt(mean_squared_log_error(X_test, pred)).round(5)

# Display RMSLE score and description
st.sidebar.subheader("Evaluation Metrics")
st.sidebar.write("RMSLE score for this lag order: ", rmsle)
st.sidebar.write("""
                 This is a measure of the difference between the predicted 
                 and actual values of the target variable. A lower score indicates 
                 better performance.
                 """)

# Generate sales forecasts and display them in a chart
forecast = result.forecast(steps=forecast_period)
st.subheader("Sales Forecast")
st.write("Sales Forecast for the next", forecast_period, 'days')
forecast_chart = pd.DataFrame({"Forecasted Sales": forecast})
st.line_chart(forecast_chart)

# Display sales data and model parameters
st.header("Forecast")
st.write("Average Sales for the next", forecast_period, "days")
st.write(sales_data)