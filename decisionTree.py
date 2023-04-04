# make this streamlit sales prediction app look more beautiful by adding a background and other styles:
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the data
data = pd.read_csv('merged_train_data.csv')

# Split the data
X = data.drop('sales', axis=1)

# Load the model and encoder ans scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


    # merged3=merged3.set_index(['date'])

# Define the function to make predictions
def predict_sales(input_data, input_df):
    # defining categories and numeric columns
    categoric_columns = ['family', 'city', 'state', 'type_y', 'type_x']
    columns = list(input_df.columns) 
    numeric_columns = [i for i in columns if i not in categoric_columns]
    scaled_num = scaler.fit_transform(input_df[numeric_columns])
    encoded_cat = encoder.transform(input_df[categoric_columns])
    input_data = pd.concat([scaled_num, encoded_cat], axis=1)
    # convert input_data to a numpy array before flattening to convert it back to a 2D array
    input_data = input_data.to_numpy()
    prediction = model.predict(input_data.flatten().reshape(1, -1))
    return prediction

# Create the app
st.set_page_config(page_title='Sales Prediction App', page_icon=':bar_chart:', layout='wide')
st.title('Sales Prediction App')

# introduction
st.write("""
This app predicts the sales amount for a given store and date based on various input features. 
Please enter the required information and click on 'Predict' to get the predicted sales amount. 
""")

# Adding sections for related porameters
st.write('## Store Information')
store_nbr = st.selectbox('Store Number', list(X['store_nbr'].unique()))
cluster = st.slider('Cluster', 1, 17)
city = st.selectbox('City', list(X['city'].unique()))
state = st.selectbox('State', list(X['state'].unique()))

st.write('## Product Information')
family = st.selectbox('Family', list(X['family'].unique()))
type_x = st.radio('Type X', list(X['type_x'].unique()))
type_y = st.radio('Type Y', list(X['type_y'].unique()))
onpromotion = st.selectbox('On Promotion', [True, False])
oil_price = st.number_input("Enter oil price", format="%.5f")

st.write('## Date Information')
# INPUT DATE, THEN USE GETDATEFEATURES TO EXTRACT ALL THE DATE INFORMATION
date = st.date_input("Pick a date")

# Convert input parameters to a pandas DataFrame
input_dict = {
   'store_nbr': store_nbr,
   'cluster': cluster,
   'city': city,
   'state': state,
   'family': family,
   'type_x': type_x,
   'type_y': type_y,
   'onpromotion': onpromotion,
   'oil_price': oil_price,
   'date' : date
  }

input_df = pd.DataFrame([input_dict])

def getDateFeatures(df):
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['week_of_year'] = df['week_of_year'].astype(float)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df["is_weekend"] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)

    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    df['pay_day'] = np.where((df['day_of_month']==15) | (df['is_month_end']==1), 1, 0)
    df['earthquake_impact'] = np.where(df['date'].isin(
        pd.date_range(start='2016-04-16', end='2016-12-31', freq='D')), 1, 0)

    return df

input_df = getDateFeatures(input_df)
input_df = input_df.drop(columns= ['date'], axis=1)

# Make prediction and show results
if st.button('Predict'):
  prediction = predict_sales(input_df.values, input_df)
  st.success('The predicted sales amount is $' + str(round(prediction[0],2)))