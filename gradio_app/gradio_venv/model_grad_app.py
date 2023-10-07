import gradio as gr 
import numpy as np 
import pandas as pd 
import pickle

#below we have our key inputs lists 
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

#below we defined some helper functions 

#below is a function to load our tools 

def find_encoder(en_path = r"gradio_app\gradio_venv\data\encoder.pkl"):
    with open(en_path, 'rb') as file:
        encoder = pickle.load(file)
    return encoder

def find_scaler(sc_path = r"gradio_app\gradio_venv\data\scaler.pkl"):
    with open(sc_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# below we load our model 
model = model() # we instantiate the mdoel 
model.load_model(r'gradio_app\gradio_venv\data\churn_best_model.pkl') 

# below we instantiate the preprocessing tools 

encoder = find_encoder()
scaler = find_scaler()