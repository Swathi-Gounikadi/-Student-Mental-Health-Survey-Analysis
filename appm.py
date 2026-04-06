import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page settings
st.set_page_config(
    page_title="🧠 Mental Health Depression Prediction"
)

# Load saved files using pickle
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Title
st.title("🧠 Mental Health Depression Prediction")
st.markdown("Fill all details below and click Predict.")

# Inputs
age = st.number_input("Age", min_value=10, max_value=100, value=25)

academic_pressure = st.selectbox(
    "Academic Pressure",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: ["None", "Low", "Slightly Low", "Moderate", "High", "Very High"][x]
)

work_pressure = st.selectbox(
    "Work Pressure",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: ["None", "Low", "Slightly Low", "Moderate", "High", "Very High"][x]
)

study_satisfaction = st.selectbox(
    "Study Satisfaction",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: [
        "Very Dissatisfied", "Dissatisfied", "Slightly Satisfied",
        "Neutral", "Satisfied", "Very Satisfied"
    ][x]
)

job_satisfaction = st.selectbox(
    "Job Satisfaction",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: [
        "Very Dissatisfied", "Dissatisfied", "Slightly Satisfied",
        "Neutral", "Satisfied", "Very Satisfied"
    ][x]
)

work_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, value=8.0)

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)

gender = st.selectbox("Gender", ["Male", "Female"])

city = st.selectbox(
    "City",
    ["Kalyan", "Srinagar", "Hyderabad", "Vasai-Virar", "Lucknow", "Other"]
)

profession = st.selectbox(
    "Profession",
    ["Student", "Teacher", "Engineer", "Doctor", "Software Developer", "Business", "Farmer", "Other"]
)

sleep = st.selectbox(
    "Sleep Duration",
    ["<5 hours", "5-6 hours", "6-7 hours", "7-8 hours", ">8 hours"]
)

diet = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)

degree = st.selectbox(
    "Degree",
    ["10th", "12th", "Diploma", "B.Tech", "B.Sc", "B.Com", "BBA", "BA",
     "M.Tech", "M.Sc", "MBA", "MCA", "PhD", "Other"]
)

suicidal = st.selectbox(
    "Have you ever had suicidal thoughts?",
    ["No", "Sometimes", "Often", "Yes"]
)

financial = st.selectbox(
    "Financial Stress",
    ["Very Low", "Low", "Moderate", "High", "Very High"]
)

family = st.selectbox(
    "Family History of Mental Illness",
    ["No", "Yes"]
)

# Prediction
if st.button("Predict"):

    input_df = pd.DataFrame([{
        'Gender': str(gender),
        'Age': float(age),
        'City': str(city),
        'Profession': str(profession),
        'Academic Pressure': float(academic_pressure),
        'Work Pressure': float(work_pressure),
        'CGPA': float(cgpa),
        'Study Satisfaction': float(study_satisfaction),
        'Job Satisfaction': float(job_satisfaction),
        'Sleep Duration': str(sleep),
        'Dietary Habits': str(diet),
        'Degree': str(degree),
        'Have you ever had suicidal thoughts ?': str(suicidal),
        'Work/Study Hours': float(work_hours),
        'Financial Stress': str(financial),
        'Family History of Mental Illness': str(family)
    }])

    categorical_cols = [
        'Gender', 'City', 'Profession', 'Sleep Duration',
        'Dietary Habits', 'Degree',
        'Have you ever had suicidal thoughts ?',
        'Financial Stress', 'Family History of Mental Illness'
    ]

    numeric_cols = [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours'
    ]

    for col in categorical_cols:
        input_df[col] = input_df[col].fillna("Missing").astype(str)

    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    input_df[numeric_cols] = input_df[numeric_cols].fillna(0)

    try:
        processed = preprocessor.transform(input_df)

        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        processed_df = pd.DataFrame(processed, columns=feature_names)

        required_features = model.get_booster().feature_names

        for col in required_features:
            if col not in processed_df.columns:
                processed_df[col] = 0

        processed_df = processed_df[required_features]

        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Depression")
            st.markdown("""
            **💡 Suggestions:**  
            - Talk to a mental health professional  
            - Sleep 7-8 hours daily  
            - Do exercise or meditation  
            - Reduce stress  
            - Stay connected with family/friends  
            """)
            st.snow()
        else:
            st.success("✅ Low Risk of Depression")
            st.markdown("""
            **💡 Suggestions:**  
            - Maintain healthy lifestyle  
            - Monitor stress levels  
            - Stay socially active  
            """)
            st.balloons()

        st.write(f"Prediction Confidence: {probability:.2%}")
        st.progress(float(probability))

    except Exception as e:
        st.error(f"Error: {e}")
        st.write(input_df)