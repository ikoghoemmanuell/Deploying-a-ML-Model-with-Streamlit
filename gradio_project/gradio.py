# Load the necessary libraries
import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the model and encoder ans scaler
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load the data
data = pd.read_csv('data.csv')

# Define the input and output interfaces for the Gradio app
input_components = [
    gr.inputs.Dropdown(choices=["Male", "Female"], label="Gender"),
    gr.inputs.Checkbox(label="Senior Citizen"),
    gr.inputs.Checkbox(label="Partner"),
    gr.inputs.Checkbox(label="Dependents"),
    gr.inputs.Number(label="Tenure (months)", min_value=0),
    gr.inputs.Radio(choices=["Yes", "No"], label="Phone Service"),
    gr.inputs.Radio(choices=["DSL", "Fiber optic", "No"], label="Internet Service"),
    gr.inputs.Radio(choices=["Yes", "No", "No internet service"], label="Online Backup"),
    gr.inputs.Radio(choices=["Yes", "No", "No internet service"], label="Tech Support"),
    gr.inputs.Dropdown(choices=["Month-to-month", "One year", "Two year"], label="Contract"),
    gr.inputs.Radio(choices=["Yes", "No"], label="Paperless Billing"),
    gr.inputs.Dropdown(choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
    gr.inputs.Number(label="Monthly Charges", min_value=0),
    gr.inputs.Number(label="Total Charges", min_value=0),
    gr.inputs.Radio(choices=["Yes", "No", "No internet service"], label="Streaming Service"),
    gr.inputs.Radio(choices=["Yes", "No", "No internet service"], label="Security Service"),
]

output_components = [
    gr.outputs.Label(label="Churn Prediction"),
]
    # Convert the input values to a pandas DataFrame with the appropriate column names
def input_df_creator(gender, senior_citizen, partner, dependents, tenure, phone_service, internet_service, online_backup, tech_support, contract, paperless_billing, payment_method, monthly_charges, total_charges, streaming_service, security_service):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [int(senior_citizen)],
        "Partner": [int(partner)],
        "Dependents": [int(dependents)],
        "tenure": [int(tenure)],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "OnlineBackup": [online_backup],
        "TechSupport": [tech_support],
        "Contract": [contract],
        "PaperlessBilling": [int(paperless_billing)],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [float(monthly_charges)],
        "TotalCharges": [float(total_charges)],
        "StreamingService": [streaming_service],
        "SecurityService": [security_service],
    })
    return input_data

# Define the function to be called when the Gradio app is run
def predict_sales(input_data, input_df):
    # defining categories and numeric columns
    categoric_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'OnlineBackup', 'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'StreamingService', 'SecurityService']
    columns = list(input_df.columns) 
    numeric_columns = [i for i in columns if i not in categoric_columns]
    scaled_num = scaler.fit_transform(input_df[numeric_columns])
    encoded_cat = encoder.transform(input_df[categoric_columns])
    input_data = pd.concat([scaled_num, encoded_cat], axis=1)
    
    # Use the pre-trained model to make a prediction
    prediction = model.predict(input_data)[0]
    
    # Convert the prediction to a human-readable format
    if prediction == 1:
        churn_prediction = "This customer is likely to churn."
    else:
        churn_prediction = "This customer is not likely to churn."
    
    return churn_prediction

# Create the Gradio interface
gr.Interface(predict_churn, inputs=input_components, outputs=output_components, title="Churn Prediction", description="Enter the customer's demographic and service usage information to predict whether they are likely to churn or not.").launch()


