import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
model = keras.models.load_model('model_churn.h5')  # Replace 'your_model_path' with the path to your model file
scaler = pickle.load(open("scaler_churn", "rb"))

min_values = {
    'CreditScore': 350,
    'Age': 18,
    'Balance': 0,
    'Tenure': 0,
    'NumOfProducts': 1,
    'HasCrCard': 0,
    'IsActiveMember': 0,
    'EstimatedSalary': 11.58,
    'Geography_France': 0,
    'Geography_Germany': 0,
    'Geography_Spain': 0
}

def make_prediction(features):
    input_data = np.array(features).reshape(1, 12)
    scaled_features = scaler.transform(input_data)

    prediction = model.predict(scaled_features)

    return prediction

st.title('Chrun Prediction')


credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=500)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0, value=0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_credit_card = st.checkbox('Has Credit Card')
is_active_member = st.checkbox('Is Active Member')
estimated_salary = st.number_input('Estimated Salary', min_value=0, value=0)
gender = st.radio('Gender', ('Male', 'Female'))  # Add Gender radio button

geography_france_min = st.number_input('Minimum Geography_France', min_value=0, value=min_values['Geography_France'])
geography_germany_min = st.number_input('Minimum Geography_Germany', min_value=0, value=min_values['Geography_Germany'])
geography_spain_min = st.number_input('Minimum Geography_Spain', min_value=0, value=min_values['Geography_Spain'])

gender = 1 if gender == 'Male' else 0

if st.button('Predict'):
    features = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': int(has_credit_card),  # Convert boolean to 1 or 0
        'IsActiveMember': int(is_active_member),  # Convert boolean to 1 or 0
        'EstimatedSalary': estimated_salary,
        'Geography_France': geography_france_min,
        'Geography_Germany': geography_germany_min,
        'Geography_Spain': geography_spain_min,
        'Gender': gender
    }
    print(features)

    
    prediction = make_prediction(list(features.values()))

    if prediction >= 0.5:
        st.write('Prediction: High Risk')
    else:
        st.write('Prediction: Low Risk')
