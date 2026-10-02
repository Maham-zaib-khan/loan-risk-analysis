import streamlit as st
import pandas as pd
import joblib

# Set up the page configuration
st.set_page_config(page_title="Loan Risk Predictor", page_icon="🏦", layout="centered")

# Load the trained model only once to save memory and speed up the app
@st.cache_resource
def load_model():
    return joblib.load('svm_loan_model.pkl')

model = load_model()

# Build the UI Header
st.title("🏦 Retail Loan Risk Assessor")
st.markdown("Enter the borrower's details below to predict if they are **Reliable** or **Unreliable**.")

# Create input fields for the user
st.subheader("Borrower Financial Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=10000, value=50000, step=5000)
    
with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=500, value=15000, step=1000)

# The predict button logic
if st.button("Evaluate Loan Application", type="primary"):
    
    # 1. Calculate the engineered feature (Loan-to-Income)
    loan_to_income = loan_amount / income
    
    # 2. Package the inputs into a DataFrame exactly as the model expects them
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score],
        'loan_to_income': [loan_to_income]
    })
    
    # 3. Make the prediction
    prediction = model.predict(input_data)[0]
    
    # 4. Display the results dynamically
    st.divider()
    if prediction == 0:
        st.success("✅ **Approved Classification: Reliable Borrower**")
        st.write("This profile matches historical patterns of successful repayment.")
    else:
        st.error("⚠️️ **Flagged Classification: Unreliable Borrower**")
        st.write("This profile indicates high default risk. Manual underwriting review recommended.")
        
    # Display the calculated LTI for the underwriter's reference
    st.caption(f"Calculated Loan-to-Income Ratio: {loan_to_income:.2f}x")