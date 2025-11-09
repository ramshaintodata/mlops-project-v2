import streamlit as st
import joblib
import pandas as pd
import numpy as np
import wandb

# -------------------------------
# 🎯 Project Info
# -------------------------------
PROJECT = "mlops-project-v2"
MODEL_NAME = "logistic-regression-tuned"
MODEL_FILE = "models/logistic_regression_tuned.pkl"

st.title("💳 Loan Status Prediction App")
st.write("This app predicts whether a loan will be approved or not using a tuned Logistic Regression model.")

# -------------------------------
# 🧠 Load Model (from local or W&B)
# -------------------------------

try:
    # Try loading model directly from local file
    model = joblib.load(MODEL_FILE)
    st.success("✅ Model loaded successfully from local directory!")

except Exception as e:
    st.warning("⚠️ Local model not found — loading from Weights & Biases (W&B)...")

    try:
        run = wandb.init(project=PROJECT, job_type="inference")
        artifact = run.use_artifact(f"{MODEL_NAME}:latest", type="model")
        artifact_dir = artifact.download()
        model = joblib.load(f"{artifact_dir}/logistic_regression_tuned.pkl")
        run.finish()
        st.success("✅ Model loaded successfully from W&B artifact!")
    except Exception as e2:
        st.error(f"❌ Failed to load model: {e2}")
        st.stop()

# -------------------------------
# 🧾 Input Fields
# -------------------------------

st.subheader("🔢 Enter Applicant Information")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
person_income = st.number_input("Monthly Income ($)", min_value=0, step=100)
person_emp_exp = st.number_input("Years of Experience", min_value=0, step=1)
person_home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage", "other"])
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, step=500)
loan_intent = st.selectbox("Loan Intent", ["education", "medical", "venture", "personal", "homeimprovement", "debtconsolidation"])
loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["yes", "no"])

# -------------------------------
# 🧮 Prepare Input Data
# -------------------------------
# --- INPUTS ---
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, step=0.01)
credit_history_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
previous_loan_defaults = st.selectbox("Previous Loan Defaults", ["yes", "no"])
home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage", "other"])
employment_type = st.selectbox("Employment Type", ["salaried", "self-employed", "unemployed", "student"])

# --- PREPROCESSING ---
# Convert categorical values to the same numeric codes used during training
default_map = {"yes": 1, "no": 0}
home_map = {"rent": 0, "own": 1, "mortgage": 2, "other": 3}
employment_map = {"salaried": 0, "self-employed": 1, "unemployed": 2, "student": 3}

person_home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage", "other"], key="person_home_ownership")
home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage", "other"], key="encoded_home_ownership")

previous_loan_defaults = st.selectbox("Previous Loan Defaults", ["yes", "no"])



# Create input DataFrame
input_data = pd.DataFrame({
    "LoanPercentIncome": [loan_percent_income],
    "CreditHistory": [credit_history_length],
    "CreditScore": [credit_score],
    "PreviousLoanDefaults": [default_map[previous_loan_defaults]],
    "HomeOwnership": [home_map[home_ownership]],
    "EmploymentType": [employment_map[employment_type]]
})

# 🔍 Prediction
# -------------------------------
if st.button("🔮 Predict Loan Status"):
    try:
        prediction = model.predict(input_data)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.subheader(f"Prediction: {result}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
