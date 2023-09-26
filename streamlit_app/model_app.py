# below we are importing the necessary packages 
import streamlit as st 
import pickle
import numpy as np 
import pandas as pd 
import os
from xgboost import XGBRegressor

# below we defined a function to load the train data set 
def train_load(relative_path):
    train_data = pd.read_csv(relative_path)
    return train_data  # You should return the loaded data, not the function itself

# below we defined a function to locate and return the path of our train set
def find_train_set(file_name, search_directory):
    for root, dirs, files in os.walk(search_directory):
        if file_name in files:
            return os.path.join(root, file_name)  # Corrected the os.path.join method
    return None

# below we are declaring variables to store both the file and the location to search
file_to_find = 'X_train.csv'
path_to_search = 'G:/AZUBI-AFRICA/CAREER_ACCELERATOR_ALL_OUT/LP3_REGR/lp3_submission_regress'

file_path = find_train_set(file_to_find, path_to_search)

# below we are calling the function to load the data set  
train_data_df = train_load(file_path)

# below are setting the title of the app 
st.title('Store Sales Regression Model App')

# below is the main content area 
Header = st.container()
Data_section  = st.container()
Prediction_section = st.container()

# below is is the app description and introduction
with Header:
    Header.write('Welcome to the Store Sales Regression Model App!')
    Header.write('This app uses machine learning to predict store sales based on historical data.')
    Header.write('To use the app, follow these steps:')
    Header.markdown('- Please enter your inputs values')
    Header.markdown('- Adjust any necessary settings or parameters.')
    Header.markdown('- Click the "Predict" button to generate sales predictions.')

# below we are setting and loading our data set 
with st.expander('Data Preview'):
    st.write('**Training Data Preview:**')
    st.write(train_data_df.head()) # allowing the user to preview the loaded train data
    st.write(f'Model Training  Data Shape: {train_data_df.shape}')



