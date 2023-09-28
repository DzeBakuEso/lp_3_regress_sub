# belowe we import our packages 

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor

# Setting the canvas (background) color to light blue
st.markdown(
    """
    <style>
    .reportview-container {
        background: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the welcome message and instructions as a multiline string
welcome_message = """
# Store Sales Regression Model App

This app uses machine learning to predict store sales based on historical data.

To use the app, follow these steps:

- Please enter your input values.
- Adjust any necessary settings or parameters.
- Click the "Predict" button to generate sales predictions.
"""

# Below we defined the app sections
header = st.container()
prediction_section = st.container()

#below is a unction to locate and return the path of the train set
def find_train_set(file_name, search_directory):
    for root, dirs, files in os.walk(search_directory):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

#below we defined function to load train data
@st.cache_data
def train_load(relative_path):
    train_data = pd.read_csv(relative_path)
    return train_data

#below we declared variables to store both the file and the location to search
file_to_find = 'X_train.csv'
path_to_search = os.path.join(os.getcwd(), 'data')

file_path = find_train_set(file_to_find, path_to_search)

# Load the data set
train_data_df = train_load(file_path)
#below we create a dropdown to toggle visibility of the training data
show_train_data = st.checkbox("Show Training Data")

#below this code displayed the loaded data if the checkbox is selected
if show_train_data:
    st.write('**Training Data Preview:**')
    st.write(train_data_df.head())  # Allow the user to preview the loaded train data
    st.write(f'Model Training Data Shape: {train_data_df.shape}')

#below is a function to load the preprocessing tools
def load_preprocessing_tools(tool_path):
    with open(tool_path, 'rb') as file:
        loaded_tools = pickle.load(file)
    return loaded_tools

#below we Defined relative paths for the preprocessing tools
data_directory = os.path.join(os.getcwd(), 'data')
encoder_path = os.path.join(data_directory, 'family_encoding.pkl')
scaler_path = os.path.join(data_directory, 'standard_scaler.pkl')

#below we Load the preprocessing tools
loaded_encoder = load_preprocessing_tools(encoder_path)
loaded_scaler = load_preprocessing_tools(scaler_path)

#below we defined a function to locate and return the path of the model
def find_model(model_name, model_directory):
    for root, dirs, files in os.walk(model_directory):
        if model_name in files:
            return os.path.join(root, model_name)
    return None

#below we decalare a variables for model name and the directory
model_name = 'xgboost_model.pkl'
model_directory = os.path.join(os.getcwd(), 'data')

model_path = find_model(model_name, model_directory)

#below is a function to load the model
def load_xgboost_model(model_path):
    with open(model_path, 'rb') as file:
        xgboost_model = pickle.load(file)
    return xgboost_model

# here we are Loading the XGBoost model
loaded_xgboost_model = load_xgboost_model(model_path)

# Displaying the welcome message and instructions
with header:
    st.markdown(welcome_message, unsafe_allow_html=True)


