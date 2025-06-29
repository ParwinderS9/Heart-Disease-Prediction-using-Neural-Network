import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go

# Load trained model and scaler
model = load_model("heart.h5")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Predict the risk of heart disease based on patient clinical features.")

# Input function
def get_user_input():
    st.sidebar.header("Patient Features")

    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.sidebar.slider('Resting BP', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 200, 300)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    restecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    thalach = st.sidebar.slider('Max Heart Rate', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.sidebar.slider('Major Vessels Colored', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Encoding
    sex = 1 if sex == 'Male' else 0
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_map[cp]
    fbs = 1 if fbs == 'Yes' else 0
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    restecg = restecg_map[restecg]
    exang = 1 if exang == 'Yes' else 0
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope = slope_map[slope]
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    thal = thal_map[thal]

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
    
    return data, {
        'Age': age, 'Resting BP': trestbps, 'Cholesterol': chol,
        'Max HR': thalach, 'Oldpeak': oldpeak
    }

# Risk classifier
def classify_risk(pred):
    if pred < 0.5:
        return "Low", "green", "✅ Low Risk Detected"
    elif pred < 0.75:
        return "Moderate", "orange", "⚠️ Moderate Risk Detected"
    else:
        return "High", "red", "🚨 High Risk Detected"

# Predict and Display
def make_prediction(data, label_data):
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)[0][0]
    risk, color, msg = classify_risk(prediction)

    st.subheader("🩺 Prediction Result")
    st.markdown(f"### {msg}")
    st.metric(label="Risk Score", value=f"{prediction:.2f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction * 100,
        title={'text': "Risk Level (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'red'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Display input comparison
    st.subheader("📊 Patient Data Overview")
    healthy_ref = {'Age': 50, 'Resting BP': 120, 'Cholesterol': 200, 'Max HR': 150, 'Oldpeak': 1.0}
    df = pd.DataFrame({
        'Feature': list(label_data.keys()),
        'Patient': list(label_data.values()),
        'Healthy': [healthy_ref[k] for k in label_data.keys()]
    })
    st.dataframe(df)

# ⬅️ Sidebar input
input_data, display_data = get_user_input()

if st.button("🔮 Predict"):
    make_prediction(input_data, display_data)

# Optional: Quick Test Samples
st.markdown("---")
st.subheader("🎯 Test Predefined Risk Levels")

def sample_case(level):
    if level == "Low":
        return [[40, 0, 0, 110, 180, 0, 0, 170, 0, 0.5, 0, 0, 1]]
    elif level == "Moderate":
        return [[55, 1, 2, 130, 240, 1, 1, 140, 1, 2.0, 1, 1, 2]]
    elif level == "High":
        return [[80, 1, 3, 180, 500, 1, 2, 60, 1, 5.0, 2, 4, 3]]

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟢 Low Risk Sample"):
        data = np.array(sample_case("Low"))
        make_prediction(data, {'Age': 40, 'Resting BP': 110, 'Cholesterol': 180, 'Max HR': 170, 'Oldpeak': 0.5})
with col2:
    if st.button("🟠 Moderate Risk Sample"):
        data = np.array(sample_case("Moderate"))
        make_prediction(data, {'Age': 55, 'Resting BP': 130, 'Cholesterol': 240, 'Max HR': 140, 'Oldpeak': 2.0})
with col3:
    if st.button("🔴 High Risk Sample"):
        data = np.array(sample_case("High"))
        make_prediction(data, {'Age': 80, 'Resting BP': 180, 'Cholesterol': 500, 'Max HR': 60, 'Oldpeak': 5.0})

st.markdown("---")
st.markdown("⚠️ *This tool is for educational purposes only. Consult a physician for medical advice.*")
