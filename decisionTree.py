# fix the errors "1 has type int, but expected one of: bytes, unicode' in the streamlit app below:
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the data
data = pd.read_csv('merged_train_data.csv')

# Split the data
X = data.drop('sales', axis=1)

# Load the model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Define the function to make predictions
def predict_sales(input_data):
    cols = ['family', 'city', 'state', 'cluster', 'type_y', 'type_x']
    for col in cols:
        input_data[col] = encoder.transform(input_data[col])
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

st.write('## Date Information')
year = st.number_input('Year')
month = st.number_input('Month', min_value=1, max_value=12)
day_of_month = st.number_input('Day of Month', min_value=1, max_value=31)
day_of_week = st.number_input('Day of Week', min_value=1, max_value=7)
day_of_year = st.number_input('Day of Year', min_value=1, max_value=365)
week_of_year = st.number_input('Week of Year', min_value=1, max_value=52)
quarter = st.number_input('Quarter', min_value=1, max_value=4)
is_weekend = st.selectbox('Is Weekend', ['True', 'False'])
is_month_start = st.selectbox('Is Month Start', ['True', 'False'])
is_month_end = st.selectbox('Is Month End', ['True', 'False'])
is_quarter_start = st.selectbox('Is Quarter Start', ['True', 'False'])
is_quarter_end = st.selectbox('Is Quarter End', ['True', 'False'])
is_year_start = st.selectbox('Is Year Start', ['True', 'False'])

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
   'year': year,
   'month': month,
   'day_of_month': day_of_month,
   'day_of_week': day_of_week,
   'day_of_year': day_of_year,
   'week_of_year': week_of_year,
   'quarter': quarter,
   'is_weekend': is_weekend,
   'is_month_start': is_month_start,
   'is_month_end': is_month_end,
   'is_quarter_start': is_quarter_start,
   'is_quarter_end': is_quarter_end,
   'is_year_start': is_year_start
  }

input_df = pd.DataFrame([input_dict])


# Make prediction and show results
if st.button('Predict'):
  prediction = predict_sales(input_df.values)
  st.success('The predicted sales amount is $' + str(round(prediction,2)))