import gradio as gr
import numpy as np
import pickle
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Define the expected input feature names (lowercase and without underscores)

expected_inputs = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48',
    'tenure_group_49 - 60', 'tenure_group_61 - 72', 'MonthlyCharges',
    'TotalCharges', 'SeniorCitizen'
]

# Define the categorical and numerical features (use lowercase and without underscores)
categoricals = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_13 - 24', 'tenure_group_25 - 36', 'tenure_group_37 - 48',
    'tenure_group_49 - 60', 'tenure_group_61 - 72', 'SeniorCitizen'
]

numericals = ['MonthlyCharges', 'TotalCharges']

# Define a function to load the encoder
def find_encoder(en_path="data/encoder.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path = os.path.join(script_dir, en_path)

    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)
    return encoder

# Define a function to load the scaler
def find_scaler(sc_path="data/scaler.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(script_dir, sc_path)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Define the path to the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_relative_path = 'data/churn_best_model.pkl'
model_path = os.path.join(script_dir, model_relative_path)

# Load the model using joblib
model = RandomForestClassifier()
model = joblib.load(model_path)

# Load the encoder and scaler
encoder = find_encoder()
scaler = find_scaler()

# Define a function to process the inputs
def process_inputs(inputs, categoricals=categoricals, numericals=numericals, encoder=encoder, scaler=scaler):
    processed_inputs = {}

    # Process categorical inputs
    for column in categoricals:
        input_value = inputs[column]
        encoded_inputs = encoder.transform([[input_value]])
        processed_inputs[column] = encoded_inputs[0][0]

    # Process numerical inputs
    for column in numericals:
        input_value = inputs[column]
        scaled_inputs = scaler.transform([[input_value]])
        processed_inputs[column] = scaled_inputs[0][0]

    return processed_inputs

# Define the function for making predictions
def predict(inputs):
    # Process the inputs
    processed_inputs = process_inputs(inputs, categoricals, numericals, encoder, scaler)
    # Make predictions using our model
    prediction = model.predict([list(processed_inputs.values())])[0]
    return "Churn: Yes" if prediction else "Churn: No"

# Define the Gradio components for input features with original variable names
gender_Male = gr.Radio(choices=['M', 'F'], label="Choose gender (Male or Female)")
Partner_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have a partner?")
Dependents_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have dependents?")
PhoneService_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have phone service?")
MultipleLines_No_phone_service = gr.Radio(choices=["Yes", "No"], label="Do you have multiple lines (No phone service)?")
MultipleLines_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have multiple lines?")
InternetService_Fiber_optic = gr.Radio(choices=["Yes", "No"], label="Do you have fiber optic internet service?")
InternetService_No = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service?")
OnlineSecurity_No_internet_service = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service for online security?")
OnlineSecurity_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have online security?")
OnlineBackup_No_internet_service = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service for online backup?")
OnlineBackup_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have online backup?")
DeviceProtection_No_internet_service = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service for device protection?")
DeviceProtection_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have device protection?")
TechSupport_No_internet_service = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service for tech support?")
TechSupport_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have tech support?")
StreamingTV_No_internet_service = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service for streaming TV?")
StreamingTV_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have streaming TV?")
StreamingMovies_No_internet_service = gr.Radio(choices=["Yes", "No"], label="Do you have no internet service for streaming movies?")
StreamingMovies_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have streaming movies?")
Contract_One_year = gr.Radio(choices=["Yes", "No"], label="Do you have a one-year contract?")
Contract_Two_year = gr.Radio(choices=["Yes", "No"], label="Do you have a two-year contract?")
PaperlessBilling_Yes = gr.Radio(choices=["Yes", "No"], label="Do you have paperless billing?")
PaymentMethod_Credit_card_automatic = gr.Radio(choices=["Yes", "No"], label="Do you use credit card (automatic) for payment?")
PaymentMethod_Electronic_check = gr.Radio(choices=["Yes", "No"], label="Do you use electronic check for payment?")
PaymentMethod_Mailed_check = gr.Radio(choices=["Yes", "No"], label="Do you use mailed check for payment?")
tenure_group_13_24 = gr.Radio(choices=["Yes", "No"], label="Is your tenure between 13 and 24 months?")
tenure_group_25_36 = gr.Radio(choices=["Yes", "No"], label="Is your tenure between 25 and 36 months?")
tenure_group_37_48 = gr.Radio(choices=["Yes", "No"], label="Is your tenure between 37 and 48 months?")
tenure_group_49_60 = gr.Radio(choices=["Yes", "No"], label="Is your tenure between 49 and 60 months?")
tenure_group_61_72 = gr.Radio(choices=["Yes", "No"], label="Is your tenure between 61 and 72 months?")
SeniorCitizen = gr.Radio(choices=["Yes", "No"], label="Are you a senior citizen?")
MonthlyCharges = gr.Number(label='What is your Monthly charge ?', minimum=0, maximum=3000, interactive=True, step=0.5, value=100)
TotalCharges = gr.Number(label='What is your Total charge ?', minimum=0, maximum=6000, interactive=True, step=0.5, value=100)

# Set up the app interface
# Create a list of input components
input_components = [
    gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes,
    MultipleLines_No_phone_service, MultipleLines_Yes,
    InternetService_Fiber_optic, InternetService_No,
    OnlineSecurity_No_internet_service, OnlineSecurity_Yes,
    OnlineBackup_No_internet_service, OnlineBackup_Yes,
    DeviceProtection_No_internet_service, DeviceProtection_Yes,
    TechSupport_No_internet_service, TechSupport_Yes,
    StreamingTV_No_internet_service, StreamingTV_Yes,
    StreamingMovies_No_internet_service, StreamingMovies_Yes,
    Contract_One_year, Contract_Two_year, PaperlessBilling_Yes,
    PaymentMethod_Credit_card_automatic,
    PaymentMethod_Electronic_check, PaymentMethod_Mailed_check,
    tenure_group_13_24, tenure_group_25_36, tenure_group_37_48,
    tenure_group_49_60, tenure_group_61_72, MonthlyCharges,
    TotalCharges, SeniorCitizen
]

# Set up the app interface
gr.Interface(
    inputs=input_components,
    outputs=gr.Label('Please Wait...'),
    fn=predict,
    title='Customer Churn Prediction',
    description="Enter customer information to predict churn."
).launch(inbrowser=True, show_error=True)





