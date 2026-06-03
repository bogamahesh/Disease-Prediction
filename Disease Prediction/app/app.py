"""
Disease Prediction Web Application
Diabetes Prediction using Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
        color: #1a1a2e !important;
    }
    .risk-high {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 3px solid #ef5350;
    }
    .risk-low {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 3px solid #43a047;
    }
    .risk-high h2, .risk-low h2 {
        color: #1a1a2e !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'best_model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# Header
st.markdown('<div class="main-header"><h1>🩺 Diabetes Prediction System</h1><p>Enter patient details to predict diabetes risk</p></div>', unsafe_allow_html=True)

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)

with col2:
    glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=300, value=120)
    insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=900, value=80)
    age = st.number_input('Age', min_value=18, max_value=100, value=30)

with col3:
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=70)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)

# Predict button
if st.button('🔍 Predict Diabetes Risk'):
    if model is None or scaler is None:
        st.error('Model not found! Please run the model building notebook first.')
    else:
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.markdown(f'''
            <div class="prediction-box risk-high">
                <h2>⚠️ High Risk of Diabetes</h2>
                <p style="font-size: 1.5rem;">Risk Score: {probability:.1%}</p>
                <p>Probability: {probability:.2%}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.warning('Please consult a healthcare professional for proper diagnosis.')
        else:
            st.markdown(f'''
            <div class="prediction-box risk-low">
                <h2>✅ Low Risk of Diabetes</h2>
                <p style="font-size: 1.5rem;">Risk Score: {probability:.1%}</p>
                <p>Probability of diabetes: {probability:.2%}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.success('Continue maintaining a healthy lifestyle!')

        # Feature importance visualization
        if hasattr(model, 'feature_importances_'):
            st.subheader('Feature Importance')
            importance = model.feature_importances_
            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            feat_df = pd.DataFrame({'Feature': features, 'Importance': importance})
            feat_df = feat_df.sort_values('Importance', ascending=True)
            st.bar_chart(feat_df.set_index('Feature'))

st.markdown('---')
st.caption('Built with Streamlit · Diabetes Dataset (PIMA Indians)')
