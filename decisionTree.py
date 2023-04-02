import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('merged_train_data.csv')

# Split the data into features and target
X = data.drop('sales', axis=1)
y = data['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Create the app
st.title('Sales Prediction App')

# Add inputs for the user to enter the values
store_nbr = st.number_input('Store Number')
family = st.selectbox('Family', list(X['family'].unique()))
onpromotion = st.selectbox('On Promotion', ['True', 'False'])
city = st.selectbox('City', list(X['city'].unique()))
state = st.selectbox('State', list(X['state'].unique()))
type_x = st.selectbox('Type X', list(X['type_x'].unique()))
cluster = st.number_input('Cluster')
oil_price = st.number_input('Oil Price')
type_y = st.selectbox('Type Y', list(X['type_y'].unique()))
month = st.number_input('Month', min_value=1, max_value=12)
day_of_month = st.number_input('Day of Month', min_value=1, max_value=31)
day_of_year = st.number_input('Day of Year', min_value=1, max_value=365)
week_of_year = st.number_input('Week of Year', min_value=1, max_value=52)
day_of_week = st.number_input('Day of Week', min_value=1, max_value=7)
year = st.number_input('Year')
is_weekend = st.selectbox('Is Weekend', ['True', 'False'])
is_month_start = st.selectbox('Is Month Start', ['True', 'False'])
quarter = st.number_input('Quarter', min_value=1, max_value=4)
is_month_end = st.selectbox('Is Month End', ['True', 'False'])
is_quarter_start = st.selectbox('Is Quarter Start', ['True', 'False'])
is_quarter_end = st.selectbox('Is Quarter End', ['True', 'False'])
is_year_start = st.selectbox('Is Year Start', ['True', 'False'])
is_year_end = st.selectbox('Is Year End', ['True', 'False'])
season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Fall'])
pay_day = st.selectbox('Pay Day', ['True', 'False'])
earthquake_impact = st.selectbox('Earthquake Impact', ['True', 'False'])

# When the 'Predict' button is clicked, make the prediction and display it
if st.button('Predict'):
    input_data = np.array([[store_nbr, family, onpromotion, city, state, type_x, cluster, oil_price, type_y, month, day_of_month, day_of_year, week_of_year, day_of_week, year, is_weekend, is_month_start, quarter, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end, season, pay_day, earthquake_impact]])
    prediction = model.predict(input_data)
    st.write('The predicted sales amount is:', prediction)
