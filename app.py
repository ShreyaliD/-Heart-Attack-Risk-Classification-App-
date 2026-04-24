import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open('model.pkl', 'rb'))

# give title
st.title("❤️ Heart Attack Risk Classification App 🩺")

# input variables
Age = st.number_input('Age', min_value=20, max_value=100, value=25)
RestingBP = st.number_input('RestingBP', min_value=0, max_value=300, value=100)
Cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=100)
FastingBS = st.selectbox('FastingBS', (0, 1))
MaxHR = st.number_input('MaxHR', min_value=60, max_value=600, value=150)
Oldpeak = st.number_input('Oldpeak', min_value=-3, max_value=10, value=2)

gender = st.selectbox('Gender', ('M', 'F'))
ChestPainType = st.selectbox('ChestPainType', ('ATA', 'NAP', 'ASY', 'TA'))
RestingECG = st.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
ExerciseAngina = st.selectbox('ExerciseAngina', ('N', 'Y'))
ST_Slope = st.selectbox('ST_Slope', ('Up', 'Flat', 'Down'))

# Encoding
Exercise_Angina = 1 if ExerciseAngina == 'Y' else 0

Sex_F = 1 if gender == 'F' else 0
Sex_M = 1 if gender == 'M' else 0

ChestPainType_dict = {'ATA': 3, 'NAP': 2, 'ASY': 1, 'TA': 0}
ChestPainType = ChestPainType_dict[ChestPainType]

Resting_ECG_dict = {'Normal': 0, 'ST': 1, 'LVH': 2}
RestingECG = Resting_ECG_dict[RestingECG]

ST_Slope_dict = {'Down': 0, 'Up': 1, 'Flat': 2}
ST_Slope = ST_Slope_dict[ST_Slope]

# create dataframe 
input_features = pd.DataFrame({
    'Age': [Age],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'MaxHR': [MaxHR],
    'Oldpeak': [Oldpeak],
    'Exercise_Angina': [Exercise_Angina],  
    'Sex_F': [Sex_F],
    'Sex_M': [Sex_M],
    'Chest_PainType': [ChestPainType],    
    'Resting_ECG': [RestingECG],         
    'st_Slope': [ST_Slope]                
})

# Scaling 
scaler = StandardScaler()
cols_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
input_features[cols_to_scale] = scaler.fit_transform(input_features[cols_to_scale])

# prediction
if st.button('Predict'):
    predictions = model.predict(input_features)

    if predictions[0] == 1:
        st.error('🚨 High Risk of Heart Attack 💔')
    else:
        st.success('✅ Low Risk of Heart Attack 💚')