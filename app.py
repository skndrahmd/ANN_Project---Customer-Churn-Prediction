import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load model
model = tf.keras.models.load_model("model.h5")

# Load scalers and encoders
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("one_hote_encoder_pickle.pkl", "rb") as file:
    geography_encoder = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User input
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", geography_encoder.categories_[0])
age = st.number_input("Age", min_value=18, max_value=99)
tenure = st.number_input("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
has_credit_card = st.checkbox("Has a Credit Card")  # Returns True or False
is_active_member = st.checkbox("Is Active Member")  # Returns True or False
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],  # Encodes gender
    "Geography": [geography],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_credit_card else 0],  # Convert checkbox to 1/0
    "IsActiveMember": [1 if is_active_member else 0],  # Convert checkbox to 1/0
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode Geography
geo_encoded = geography_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geography_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded data with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Drop the original Geography column (since it's been encoded)
input_data = input_data.drop('Geography', axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict customer churn
prediction = model.predict(input_data_scaled)[0][0]

# Output the result
if prediction > 0.5:
    st.write("The customer is likely to leave the bank")
else: 
    st.write("The customer is not likely to leave the bank")
