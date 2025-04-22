import streamlit as st
import pandas as pd
from predictor import LoanStatusPredictor 

predictor = LoanStatusPredictor(
    model='tuned_xgb_model.pkl',
    columns='final_columns.pkl',
    label_encoder='label_enc.pkl',
    onehot_encoder='onehot_enc.pkl',
    ordinal_encoder='ordinal_enc.pkl'
)

st.title("💰 Loan Status Predictor 📊")

st.write(
    "This app predicts whether your loan application will be approved or rejected based on the personal and financial information you provide.\n"
    "Simply fill in your details and get a quick prediction!"
)

st.subheader("Input Data")

with st.form(key='input_form'):
    st.subheader("Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=144, value=25)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    person_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)
    person_emp_exp = st.number_input("Employment Experience (years)", min_value=0.0, value=2.0)

    st.subheader("Loan Information")
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, value=10000.0)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"])
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.25)

    st.subheader("Credit Information")
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, value=5.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["No", "Yes"])

    submitted = st.form_submit_button("Submit")

if submitted:
    input_df = pd.DataFrame([{
            'person_age': person_age,
            'person_gender': person_gender,
            'person_education': person_education,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_home_ownership': person_home_ownership,
            'loan_amnt': loan_amnt,
            'loan_intent': loan_intent,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }])
    prediction = predictor.predict(input_df)[0]
    label = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.success(f"Your loan status is predicted to be {label}")
