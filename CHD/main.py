import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# st.write('This is a sample text')
cb_model = pickle.load(open('cb_model.pkl', 'rb'))
default = pickle.load(open('default_dict.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
import pandas as pd
# st.write(default)
# import streamlit as st
import numpy as np

st.subheader('Coronary Heart Disease predictor')
st.caption(r'This model predicts if you have chances of Coronart heart disease with a maximum accuracy of 91.8% and 100% recall')

st.info(r'Please fill the form with all the information available to you (the information you are unaware of will be replaced by the most common value of the feature in the Z-Alizadeh Sani dataset is publicly available in the UCI Machine Learning repository')
st.warning(r'Please read all the inputs carefully')
# Define the features with their full forms
features_full_forms = {
    'Age': 'Age',
    'Weight': 'Weight',
    'Length': 'Length',
    'Sex': 'Gender (Male/Female)',
    'BMI': 'Body Mass Index (BMI)',
    'DM': 'Diabetes Mellitus (DM)',
    'HTN': 'Hypertension (HTN)',
    'Current Smoker': 'Current Smoker',
    'EX-Smoker': 'Ex-Smoker',
    'FH': 'Family History of Heart Disease (FH)',
    'Obesity': 'Obesity',
    'CRF': 'Chronic Renal Failure (CRF)',
    'CVA': 'Cerebrovascular Accident (CVA)',
    'Airway disease': 'Airway Disease',
    'Thyroid Disease': 'Thyroid Disease',
    'CHF': 'Congestive Heart Failure (CHF)',
    'DLP': 'Dyslipidemia (DLP)',
    'BP': 'Blood Pressure (BP)',
    'PR': 'Pulse Rate (PR)',
    'Edema': 'Edema',
    'Weak Peripheral Pulse': 'Weak Peripheral Pulse',
    'Lung rales': 'Lung Rales',
    'Systolic Murmur': 'Systolic Murmur',
    'Diastolic Murmur': 'Diastolic Murmur',
    'Typical Chest Pain': 'Typical Chest Pain',
    'Dyspnea': 'Dyspnea',
    'Function Class': 'Functional Class',
    'Atypical': 'Atypical Chest Pain',
    'Nonanginal': 'Nonanginal Chest Pain',
    'Exertional CP': 'Exertional Chest Pain',
    'LowTH Ang': 'Low Threshold Angina (LowTH Ang)',
    'Q Wave': 'Q Wave',
    'St Elevation': 'ST Elevation',
    'St Depression': 'ST Depression',
    'Tinversion': 'T Wave Inversion',
    'LVH': 'Left Ventricular Hypertrophy (LVH)',
    'Poor R Progression': 'Poor R Progression',
    'FBS': 'Fasting Blood Sugar (FBS)',
    'CR': 'Creatinine (CR)',
    'TG': 'Triglycerides (TG)',
    'LDL': 'Low-Density Lipoprotein (LDL)',
    'HDL': 'High-Density Lipoprotein (HDL)',
    'BUN': 'Blood Urea Nitrogen (BUN)',
    'ESR': 'Erythrocyte Sedimentation Rate (ESR)',
    'HB': 'Hemoglobin (HB)',
    'K': 'Potassium (K)',
    'Na': 'Sodium (Na)',
    'WBC': 'White Blood Cell Count (WBC)',
    'Lymph': 'Lymphocyte Count (Lymph)',
    'Neut': 'Neutrophil Count (Neut)',
    'PLT': 'Platelet Count (PLT)',
    'EF-TTE': 'Ejection Fraction Measured by Transthoracic Echocardiogram (EF-TTE)',
    'Region RWMA': 'Regional Wall Motion Abnormality (Region RWMA)',
    'VHD': 'Valvular Heart Disease (VHD)'
}

user_input = {}

# st.write(len(features_full_forms.keys()))
for feature, full_form in features_full_forms.items():
    # st.write(f"### {full_form}")
    if feature == 'Sex':
        user_input[feature]= st.radio("Select gender:", ('Male', 'Female'))
    elif feature == 'Function Class':
        user_input[feature]= st.selectbox("Select functional class:", range(1, 5))
    elif 'Current Smoker' in feature:
        user_input[feature]= st.radio("Are you currently a smoker?", ('Yes', 'No'), key=f"{feature}_current")
    elif 'EX-Smoker' in feature:
        user_input[feature]= st.radio("Are you an ex-smoker?", ('Yes', 'No'), key=f"{feature}_ex")
    elif feature == 'Typical Chest Pain':
        user_input[feature]= st.radio("Do you have a Typical Chest Pain?", ('Yes', 'No'),index = 0)
    elif feature == 'DM':
        user_input[feature]= st.radio("Do you have Diabetes Mellitus?", ('Yes', 'No'),index = 1)
    elif feature == 'HTN':
        user_input[feature]= st.radio("Do you have Hypertension?", ('Yes', 'No'))

    elif feature == 'CRF':
        user_input[feature]= st.radio("Do you suffer from Chronic Renal Failure (CRF)?", ('Yes', 'No'),index = 1)
    elif feature == 'FH':
        user_input[feature]= st.radio("Do you have family history of heart diseases?", ('Yes', 'No'),index = 1)
    elif feature == 'Thyroid Disease':
        user_input[feature]= st.radio("Do you have Thyroid?", ('Yes', 'No'),index = 1)
    elif feature == 'CVA':
        user_input[feature]= st.radio("Do you suffer from a Cerebrovascular Accident?", ('Yes', 'No'),index = 1)
    elif feature == 'Airway disease':
        user_input[feature]= st.radio("Do you have Airway disease?", ('Yes', 'No'),index = 1)
    elif feature == 'Edema':
        user_input[feature]= st.radio("Do you have Edema?", ('Yes', 'No'),index = 1)
    elif feature == 'CHF':
        user_input[feature]= st.radio("Do you suffer from Congestive Heart Failure?", ('Yes', 'No'),index = 1)
    elif feature == 'DLP':
        user_input[feature]= st.radio("Do you have Dyslipidemia (DLP)?", ('Yes', 'No'),index = 1)
    elif feature == 'Weak Peripheral Pulse':
        user_input[feature]= st.radio("Do you have Weak Peripheral Pulse?", ('Yes', 'No'),index = 1)
    elif feature == 'Lung rales':
        user_input[feature]= st.radio("Do you have Lung rales?", ('Yes', 'No'),index = 1)
    elif feature == 'Systolic Murmur':
        user_input[feature]= st.radio("Do you have Systolic Murmur?", ('Yes', 'No'),index = 1)
    elif feature == 'Diastolic Murmur':
        user_input[feature]= st.radio("Do you have Diastolic Murmur?", ('Yes', 'No'),index = 1)
    elif feature == 'Dyspnea':
        user_input[feature]= st.radio("Do you have Dyspnea", ('Yes', 'No'),index = 1)
    elif feature == 'Atypical':
        user_input[feature]= st.radio("Do you have Atypical Chest pain?", ('Yes', 'No'),index = 1)
    elif feature == 'Nonanginal':
        user_input[feature]= st.radio("Do you have Nonanginal chest pain?", ('Yes', 'No'),index = 1)
    elif feature == 'Exertional CP':
        user_input[feature]= st.radio("Do you have Exertional Chest pain?", ('Yes', 'No'),index = 1)
    elif feature == 'LowTH Ang':
        user_input[feature]= st.radio("Do you have Low Threshold Angina?", ('Yes', 'No'),index = 1)
    elif feature == 'Poor R Progression':
        user_input[feature]= st.radio("Do you have Poor R Progression", ('Yes', 'No'),index = 1)
    elif feature == 'LVH':
        user_input[feature]= st.radio("Do you have Left Ventricular Hypertrophy (LVH)?", ('Yes', 'No'),index = 1)
    elif feature == 'VHD':
        user_input[feature]= st.radio("Do you have Valvular Heart Disease (VHD)?", ('Yes', 'No'),index = 1)
    elif feature == 'Obesity':
        pass
    elif feature == 'BMI':
        pass
    elif feature == 'Q Wave':
        user_input[feature]= q_wave = st.radio("Do you have Q Wave?", ('Yes', 'No'), index=1)  # Default active is 'Yes'
    elif feature == 'St Elevation':
        user_input[feature]= st_elevation = st.radio("Do you have ST Elevation?", ('Yes', 'No'), index=1)  # Default active is 'Yes'
    elif feature == 'St Depression':
        user_input[feature]= st_depression = st.radio("Do you have ST Depression?", ('Yes', 'No'), index=1)  # Default active is 'Yes'
    elif feature == 'Tinversion':
        user_input[feature]= tinversion = st.radio("Do you have T Wave Inversion?", ('Yes', 'No'), index=1)  # Default active is 'Yes'
    elif feature == 'Region RWMA':
        user_input[feature]= region_rwma = st.radio("Do you have Regional Wall Motion Abnormality (Region RWMA)?", ('Yes', 'No'), index=1)  # Default active is 'No'
    elif feature == 'CR':
        user_input[feature]= cr = st.radio("Do you have Creatinine (CR)?", ('Yes', 'No'), index=0)  # Default active is 'No'

    else:
        default_value = default[feature]
        user_input[feature] = st.number_input(f"Enter {full_form}:",value=float(default_value), step=1.0, format='%f')
# st.write(user_input)
if st.button('Submit'):
    user_input['BMI'] = user_input['Weight']//user_input['Length']
    user_input['Obesity'] = user_input['BMI'] >=30
    st.write("User input collected successfully!")
    # st.write(user_input)
    for i,j in user_input.items():
        if j == 'Yes':
            user_input[i] = 0
        elif j == 'No':
            user_input[i] = 1
    # Numerical variables:

    input_df = pd.DataFrame([user_input])
    num_cols = ['Age','Weight', 'Length','BMI', 'BP', 'PR', 'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC','Lymph', 'Neut', 'PLT', 'EF-TTE']

# Categorical variables:
    cat_cols = ['Sex', 'DM', 'HTN', 'Current Smoker', 'EX-Smoker', 'FH', 'Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease','CHF', 'DLP', 'Edema', 'Weak Peripheral Pulse', 'Lung rales', 'Systolic Murmur', 'Diastolic Murmur', 'Typical Chest Pain','Dyspnea', 'Atypical', 'Nonanginal', 'Exertional CP', 'LowTH Ang', 'Q Wave', 'St Elevation', 'St Depression', 'Tinversion','LVH', 'Poor R Progression','Cath']
    cat_cols.remove('Cath')

    # Ordinal variables
    ord_cols = ['Function Class', "Region RWMA", "VHD"]
    
    dummy_variables = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(input_df[num_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=num_cols)
    res = cb_model.predict(scaled_numerical_df.iloc[0])
    st.write(res)

