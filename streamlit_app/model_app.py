#below we are Importing the necessary packages
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor

#below we defined a function to laod look for and load the training data set 
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

# Define the welcome message and instructions as a multiline string
welcome_message = """
# Store Sales Regression Model App

This app uses machine learning to predict store sales based on historical data.

To use the app, follow these steps:

- Please enter your input values.
- Adjust any necessary settings or parameters.
- Click the "Predict" button to generate sales predictions.
"""
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

#below we defined the app section 
header = st.container()
data_set = st.container()
prediction_section = st.container()

# below we setup the header sectiion of the app 
with header:
    header.write('This app uses machine learning to predict store sales based on historical data')
    header.write('---')

#below we setup the data section of the app 
with data_set:
    if data_set.checkbox('preview the train data set'):
        data_set.write(train_data_df.head())
        data_set.markdown('please check the sidebar for more information on the dataset')    
        data_set.write('---')

#below we setupt the sidebar of the app 

# Define your data dictionary
data_dictionary = {
    'family_BABY CARE': 'Description of family_BABY CARE.',
    'family_BEAUTY': 'Description of family_BEAUTY.',
    'family_BEVERAGES': 'Description of family_BEVERAGES.',
    'family_BOOKS': 'Description of family_BOOKS.',
    'family_BREAD/BAKERY': 'Description of family_BREAD/BAKERY.',
    'family_CELEBRATION': 'Description of family_CELEBRATION.',
    'family_MAGAZINES': 'Description of family_MAGAZINES.',
    'family_MEATS': 'Description of family_MEATS.',
    'family_PERSONAL CARE': 'Description of family_PERSONAL CARE.',
    'family_PET SUPPLIES': 'Description of family_PET SUPPLIES.',
    'family_PLAYERS AND ELECTRONICS': 'Description of family_PLAYERS AND ELECTRONICS.',
    'family_POULTRY': 'Description of family_POULTRY.',
    'family_PREPARED FOODS': 'Description of family_PREPARED FOODS.',
    'family_PRODUCE': 'Description of family_PRODUCE.',
    'family_SCHOOL AND OFFICE SUPPLIES': 'Description of family_SCHOOL AND OFFICE SUPPLIES.',
    'family_SEAFOOD': 'Description of family_SEAFOOD.'
}
st.sidebar.header('Data Dictionary')
#st.sidebar.text()
# Display the data dictionary in the sidebar
for column, description in data_dictionary.items():
    st.sidebar.markdown(f"**{column}:** {description}")

#below we defined a form in the app 
form = st.form(key= "Information", clear_on_submit=True)

#below we create three separate lists two store the expected_inpputs, categorical_columns and the numerical_columns
expected_inputs = [
    'onpromotion', 'day_of_week', 'lag_1', 'rolling_mean', 'family_BABY CARE', 
    'family_BEAUTY', 'family_BEVERAGES', 'family_BOOKS', 'family_BREAD/BAKERY', 
    'family_CELEBRATION', 'family_MAGAZINES', 'family_MEATS', 'family_PERSONAL CARE', 
    'family_PET SUPPLIES', 'family_PLAYERS AND ELECTRONICS', 'family_POULTRY', 
    'family_PREPARED FOODS', 'family_PRODUCE', 'family_SCHOOL AND OFFICE SUPPLIES', 
    'family_SEAFOOD'
]

numerical_inputs = ['onpromotion', 'day_of_week', 'lag_1', 'rolling_mean']

categorical_inputs = [
    'family_BABY CARE', 'family_BEAUTY', 'family_BEVERAGES', 'family_BOOKS', 'family_BREAD/BAKERY',
    'family_CELEBRATION', 'family_MAGAZINES', 'family_MEATS', 'family_PERSONAL CARE',
    'family_PET SUPPLIES', 'family_PLAYERS AND ELECTRONICS', 'family_POULTRY',
    'family_PREPARED FOODS', 'family_PRODUCE', 'family_SCHOOL AND OFFICE SUPPLIES', 'family_SEAFOOD'
]
# Below we want to setup the prediction section of the app
with prediction_section:
    prediction_section.subheader('Inputs')
    prediction_section.write('This section will accept your inputs')
    group_one_inputs, group_two_inputs = prediction_section.columns(2)

    # Below we setup a form to accept the individual groups 
    with form:
        group_one_inputs.write('group one inputs:')
        onpromotion = group_one_inputs.number_input('Enter whether the item is on promotion or not', value=0)
        day_of_week = group_one_inputs.number_input('Please enter the day of the week', value=1, step=1)
        lag_1 = group_one_inputs.number_input('Please enter the lag_1', min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        rolling_mean = group_one_inputs.number_input('Please enter the rolling_mean', min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# below we setupt the group two inputs 
    with form:
        group_two_inputs.write('Group two inputs:')
        categorical_columns = [
    'family_BABY CARE',
    'family_BEAUTY',
    'family_BEVERAGES',
    'family_BOOKS',
    'family_BREAD/BAKERY',
    'family_CELEBRATION',
    'family_MAGAZINES',
    'family_MEATS',
    'family_PERSONAL CARE',
    'family_PET SUPPLIES',
    'family_PLAYERS AND ELECTRONICS',
    'family_POULTRY',
    'family_PREPARED FOODS',
    'family_PRODUCE',
    'family_SCHOOL AND OFFICE SUPPLIES',
    'family_SEAFOOD'
]

default_value = False  # Set your default value here as False (0)
input_value = {}  # Initialize a dictionary to store input values

if group_two_inputs.checkbox("Group two inputs"):
    for column in categorical_columns:
        input_value = group_two_inputs.checkbox(f'Include {column}', value=default_value)

# Initialize the input dictionary
input_groups_dic = {}

# Group one inputs
input_groups_dic['onpromotion'] = [onpromotion]
input_groups_dic['day_of_week'] = [day_of_week]
input_groups_dic['lag_1'] = [lag_1]
input_groups_dic['rolling_mean'] = [rolling_mean]

# Group two inputs (categorical columns)
for column in categorical_columns:
    input_groups_dic[column] = [input_value]


#below we are creating a submit button 
submitted = form.form_submit_button('submit')

#below we handle or process what happens when the submit button is clicked 
if submitted:
    with prediction_section:
       # below we format all inputs
        if submitted:
            with prediction_section:
                # Format the inputs using loaded preprocessing tools
                input_groups_dic_formatted = {}

                # Format numerical inputs using the loaded scaler
                for numerical_column in numerical_inputs:
                    input_data = np.array(input_groups_dic[numerical_column]).reshape(-1, 1)
                    scaled_input = loaded_scaler.transform(input_data)
                    input_groups_dic_formatted[numerical_column] = scaled_input[0][0]

                # Format categorical inputs using the loaded encoder
                for categorical_column in categorical_inputs:
                    input_data = np.array(input_groups_dic[categorical_column]).reshape(-1, 1)
                    encoded_input = loaded_encoder.transform(input_data)
                    input_groups_dic_formatted[categorical_column] = encoded_input[0][0]

# bewlow we want to make predictions 

# Make predictions using the loaded XGBoost model
                input_features = [input_groups_dic_formatted[column] for column in expected_inputs]
                input_features = np.array(input_features).reshape(1, -1)  # Reshape to match the model's input shape
                prediction = loaded_xgboost_model.predict(input_features)

                # Display the prediction
                st.subheader('Sales Prediction')
                st.write(f'The predicted store sales value is: {prediction[0]}')