import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor

# here we are setting the canvas (background) color to light blue
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

# Below we defined the title with a red color using Markdown
st.markdown("<h1 style='color: lightgoldenrodyellow;'>Store Sales Regression Model App</h1>", unsafe_allow_html=True)

# below we defined the app sections
Header = st.container()
Prediction_section = st.container()

#the function here is to locate and return the path of our train set
def find_train_set(file_name, search_directory):
    for root, dirs, files in os.walk(search_directory):
        if file_name in files:
            return os.path.join(root, file_name)  
    return None

#below here we defined a function to load our train data
@st.cache_data
def train_load(relative_path):
    train_data = pd.read_csv(relative_path)
    return train_data

# below we declared variables to store both the file and the location to search
file_to_find = 'X_train.csv'
path_to_search = 'G:/AZUBI-AFRICA/CAREER_ACCELERATOR_ALL_OUT/LP3_REGR/lp3_submission_regress'

file_path = find_train_set(file_to_find, path_to_search)

#below here we call the function to load the data set
train_data_df = train_load(file_path)

#the code below is to describe the app and provide an introduction
with Header:
    st.write('Welcome to the Store Sales Regression Model App!')
    st.write('This app uses machine learning to predict store sales based on historical data.')
    st.write('To use the app, follow these steps:')
    st.markdown('- Please enter your inputs values', unsafe_allow_html=True)
    st.markdown('- Adjust any necessary settings or parameters', unsafe_allow_html=True)
    st.markdown('- Click the "Predict" button to generate sales predictions', unsafe_allow_html=True)

# below here we load the data set and preview it
with st.expander('Data Preview'):
    st.write('**Training Data Preview:**')
    st.write(train_data_df.head())  # Allowing the user to preview the loaded train data
    st.write(f'Model Training Data Shape: {train_data_df.shape}')

#below here we defined a function to load the preprocessing tools
def load_preprocessing_tools(tool_path):
    with open(tool_path, 'rb') as file:
        loaded_tools = pickle.load(file)
    return loaded_tools

#here we specified the full file paths to the tool files
encoder_path = 'G:/AZUBI-AFRICA/CAREER_ACCELERATOR_ALL_OUT/LP3_REGR/lp3_submission_regress/lp_3_regress_sub/family_encoding.pkl'
scaler_path = 'G:/AZUBI-AFRICA/CAREER_ACCELERATOR_ALL_OUT/LP3_REGR/lp3_submission_regress/lp_3_regress_sub/standard_scaler.pkl'

#below here we Load the preprocessing tools
loaded_encoder = load_preprocessing_tools(encoder_path)
loaded_scaler = load_preprocessing_tools(scaler_path)

#Below we want to import our model

def find_model(model_name, model_directory):
    for root, dirs, files in os.walk(model_directory): # here we defined a function to look for the path of the model
        if model_name in files:
            return os.path.join(root, model_name)
    return None

model_name = 'xgboost_model.pkl'
model_directory = 'G:/AZUBI-AFRICA/CAREER_ACCELERATOR_ALL_OUT/LP3_REGR/lp3_submission_regress/lp_3_regress_sub/'

model_path = find_model(model_name, model_directory ) # here we call the function

# below is a function to load the our model using the path from the above model finding code 

def load_xgboost_model(model_path):
    with open(model_path, 'rb') as file:
        xgboost_model = pickle.load(file)
    return xgboost_model

loaded_xgboost_model = load_xgboost_model(model_path) # here we call and pass in the path as argument to load our model 


