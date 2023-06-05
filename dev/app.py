# Loading key libraries
import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import re
from pathlib import Path
from PIL import Image
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import StandardScaler

# Setting the page configurations
st.set_page_config(page_title= "Sales Prediction Forecasting", page_icon= ":heavy_dollar_sign:", layout= "wide", initial_sidebar_state= "auto")

# Setting the page title
st.title("Grocery Store Sales Time Series Model Prediction")




# Function to load the dataset
@st.cache_resource
def load_data(relative_path):
   data= pd.read_csv(relative_path, index_col= 0)
   #merged["date"] = pd.to_datetime(merged["date"])
   return data

    


# Loading the base dataframe
rpath = r"merged_train_data.csv"
data = load_data(rpath)



# Load the model and encoder ans scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# main sections of the app
header = st.container()
dataset = st.container()
features_and_output = st.container()




# Designing the sidebar
st.sidebar.header("Brief overview of the Columns")
st.sidebar.markdown(""" 
                    - **store_nbr** identifies the store at which the products are sold.
                    - **family** identifies the type of product sold.
                    - **sales** is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **onpromotion** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **date** is the date on which a transaction / sale was made
                    - **city** is the city in which the store is located
                    - **state** is the state in which the store is located
                    - **store_type** is the type of store, based on Corporation Favorita's own type system
                    - **cluster** is a grouping of similar stores.
                    - **oil_price** is the daily oil price
                    """)

# Structuring the dataset section
with dataset:
    if dataset.checkbox("Preview the dataset"):
        dataset.write(data.head())
        dataset.write("Further information will preview when take a look at the  sidebar")
    dataset.write("---")




# Icon for the page
image = Image.open(r"beautiful image.png")

# inputs from the user
form = st.form(key="information", clear_on_submit=True)

# Structuring the header section
with header:
    header.write("This an application to build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores")

    header.image(image)
   
    header.write("---")


 

# Structuring the features and output section
with features_and_output:
    features_and_output.subheader("Inputs")
    features_and_output.write("This section captures your input to be used in predictions")

    left_col, mid_col, right_col = features_and_output.columns(3)

    # Designing the input section of the app
    with form:
        left_col.markdown("***Combined data on Product and Transaction***")
        date = left_col.date_input("Select a date:")
        family = left_col.selectbox("Product family:", options= sorted(list(data["family"].unique())))
        onpromotion = left_col.number_input("Number of products on promotion:", min_value= data["onpromotion"].min(), value= data["onpromotion"].min())
        city = left_col.selectbox("City:", options= sorted(set(data["city"])))
    
        mid_col.markdown("***Data on Location and type***")
        store_nbr = mid_col.selectbox("Store number:", options= sorted(set(data["store_nbr"])))
        type_x = mid_col.radio("type_x:", options= sorted(set(data["type_x"])), horizontal= True)
        type_y = mid_col.radio("type_y:", options= sorted(set(data["type_y"])), horizontal= True)
        cluster = mid_col.select_slider("Store cluster:", options= sorted(set(data["cluster"])))
        state = mid_col.selectbox("State:", options= sorted(set(data["state"])))       
    
        right_col.markdown("***Data on Economical Factors***")
        oil_price = right_col.number_input("Oil price:", min_value= data["oil_price"].min(), value= data["oil_price"].min())
        
         # Submission point
        submitted = form.form_submit_button(label= "Submit button")

        if submitted:
                 with features_and_output:
                    input_features = {
                       "date":[date],
                       "store_nbr": [store_nbr],
                       "family": [family],
                       "onpromotion": [onpromotion],
                       "city": [city],
                       "state": [state],
                       "type_x": [type_x],
                       "cluster":[cluster],
                       "oil_price": [oil_price],
                       "type_y": [type_y],
                    }
                                  


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

 #Convert input parameters to a pandas DataFrame
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


@st.cache_resource
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
  
  
  # ----- Defining and structuring the footer
footer = st.expander("**Subsequent Information**")
with footer:
    if footer.button("Special Thanks"):
        footer.markdown("*We want to express our appreciation and gratitude to Emmanuel,Racheal, Mavies and Richard for their great insights and contributions!*")