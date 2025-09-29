# 💰 Loan Status Predictor 📊 

![Python](https://img.shields.io/badge/language-Python-blue)
![Streamlit](https://img.shields.io/badge/output-Streamlit-orange)

## Overview
This repository contains a **loan status prediction app** developed as a mid-exam project for the course *Model Deployment*.  
The app predicts whether a loan application will be **approved or rejected** based on personal and financial information provided by the user.

The prediction model uses an **XGBoost classifier** with proper preprocessing pipelines (encoding, missing value imputation, feature alignment).

The app is deployed using **Streamlit**, allowing users to interactively input data and get predictions without running code locally.

Live demo of the app: 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://2702346361uts.streamlit.app/)

---

## Repository Structure  
- `Dataset_A_loan.csv` → original raw dataset
- **scripts/** → Python scripts for deployment and prediction
  - `Loan_Status.ipynb` → EDA, preprocessing, and model exploration
  - `predictor.py` → contains `LoanStatusPredictor` class handling preprocessing & prediction  
  - `streamlit_app.py` → Streamlit frontend app  
- **models/** → pretrained models and encoders  
  - `tuned_xgb_model.pkl`, `ordinal_enc.pkl`, `gender_enc.pkl`, `default_enc.pkl`, `onehot_enc.pkl`, `final_columns.pkl`
- **results/**
  - `Test case.pdf` → screenshot demonstrating app functionality  
  - **Project Explanation Video** → [View on OneDrive](https://binusianorg-my.sharepoint.com/personal/syalista_nadira_binus_ac_id/_layouts/15/guestaccess.aspx?share=EVciSbCnBBdJqitGF4oBebkBZY9VrF5F4CTFWNgDu2QmyQ&e=GQCvBA)
- **requirements.txt** → dependencies for deployment

---

## Methodology
1. **Data Preprocessing**
   - Clean and standardize columns (e.g., `person_gender`, missing values)
   - Encode categorical variables:
     - `OrdinalEncoder` for hierarchical categories (education)
     - `LabelEncoder` for binary categories (gender, previous defaults)
     - `OneHotEncoder` for nominal categories (loan intent, home ownership)
   - Align features with trained model (`final_columns.pkl`)  

2. **Exploratory Data Analysis (EDA)**
   - Check distributions of numerical & categorical variables  
   - Identify missing values and outliers  

3. **Model Training**
   - Random Forest & XGBoost classifiers tested  
   - XGBoost chosen for better handling of imbalanced classes  
   - Tuned hyperparameters using `GridSearchCV`  

4. **Deployment**
   - Model wrapped in `LoanStatusPredictor` class  
   - Streamlit frontend handles user input and outputs prediction

---

## Key Insights

- XGBoost model handles imbalanced loan approval data effectively, giving better recall and F1 scores than Random Forest.  
- Proper encoding of categorical variables and handling missing values ensures reliable predictions.  
- Provides an interactive interface for real-time loan status prediction based on user inputs.  
- App clearly indicates whether a loan application is predicted as **Approved ✅** or **Rejected ❌**, making results interpretable for end-users.

---

## Author
Syalista Galuh Nadira
