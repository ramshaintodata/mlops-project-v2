import streamlit as st
import joblib
import pandas as pd
import wandb
import os

# -------------------------------
# 🎯 Project Configuration
# -------------------------------
PROJECT = "mlops-project-v2"
MODEL_NAME = "logistic-regression-tuned"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.title("💳 Loan Status Prediction App")
st.write("Predict whether a loan will be approved using the tuned Logistic Regression model.")

# -------------------------------
# 🧠 Load Model and Preprocessor
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_tuned.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        st.success("✅ Model and preprocessor loaded successfully from local directory.")
        return model, scaler, feature_names
    except Exception as e:
        st.warning(f"⚠️ Local files not found or failed to load: {e}")
        return None, None, None

model, scaler, feature_names = load_model()

# -------------------------------
# ☁️ Fallback: Load from W&B if missing
# -------------------------------
if model is None:
    st.info("Attempting to load from Weights & Biases (W&B)...")
    try:
        run = wandb.init(project=PROJECT, job_type="inference")
        artifact = run.use_artifact(f"{MODEL_NAME}:latest", type="model")
        artifact_dir = artifact.download()
        model = joblib.load(os.path.join(artifact_dir, "logistic_regression_tuned.pkl"))
        scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(artifact_dir, "feature_names.pkl"))
        run.finish()
        st.success("✅ Model and preprocessor loaded successfully from W&B.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# -------------------------------
# 🧾 Input Section
# -------------------------------
st.subheader("🧍 Applicant Information")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Monthly Income ($)", min_value=0, step=100, value=5000)
person_emp_exp = st.number_input("Years of Employment", min_value=0, max_value=50, value=3)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, step=500, value=10000)
loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

st.subheader("🏠 Other Details")

person_home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage", "other"], key="home")
loan_intent = st.selectbox("Loan Intent", ["education", "medical", "venture", "personal", "homeimprovement", "debtconsolidation"], key="intent")
previous_loan_defaults = st.selectbox("Previous Loan Defaults", ["yes", "no"], key="defaults")
employment_type = st.selectbox("Employment Type", ["salaried", "self-employed", "unemployed", "student"], key="employment")

# -------------------------------
# 🔢 Preprocessing Input
# -------------------------------
default_map = {"yes": 1, "no": 0}
home_map = {"rent": 0, "own": 1, "mortgage": 2, "other": 3}
intent_map = {
    "education": 0,
    "medical": 1,
    "venture": 2,
    "personal": 3,
    "homeimprovement": 4,
    "debtconsolidation": 5,
}
employment_map = {"salaried": 0, "self-employed": 1, "unemployed": 2, "student": 3}

input_data = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_emp_exp": [person_emp_exp],
    "person_home_ownership": [home_map[person_home_ownership]],
    "loan_amnt": [loan_amnt],
    "loan_intent": [intent_map[loan_intent]],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    "credit_score": [credit_score],
    "previous_loan_defaults_on_file": [default_map[previous_loan_defaults]],
    "employment_type": [employment_map[employment_type]]
})

# Ensure feature order consistency
if feature_names is not None:
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

# -------------------------------
# 🔮 Prediction
# -------------------------------
if st.button("🔍 Predict Loan Status"):
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.subheader(f"Prediction Result: {result}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
