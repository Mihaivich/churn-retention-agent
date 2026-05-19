import streamlit as st
import pandas as pd
import json
import boto3

# --- AWS SageMaker Configuration ---
# Ensure the endpoint name and region match your deployment
ENDPOINT_NAME = "churn-predictor-endpoint"
REGION = "us-east-1"
# -------------------------

# Initialize the Boto3 SageMaker runtime client
try:
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
except Exception as e:
    st.error(f"AWS Boto3 client initialization failed: {e}")
    st.stop()


# --- Web Application UI ---

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("👨‍💼 Customer Churn Prediction App")
st.write("This is a real-time prediction app based on a machine learning model deployed on AWS SageMaker. Please enter the customer features in the sidebar on the left and click the predict button.")

# Use the sidebar to collect user inputs for a cleaner main interface
st.sidebar.header("👤 Customer Feature Input")

# Create the input widgets
# For this demo, we only create interactive widgets for a few key features.
# In a real application, you could create widgets for all necessary features.

tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=1, step=1)
contract = st.sidebar.selectbox("Contract Type", ('Month-to-month', 'One year', 'Two year'))
internet_service = st.sidebar.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=70.7, step=0.05)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=70.7)
payment_method = st.sidebar.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

# --- Prediction Logic ---

if st.sidebar.button("🚀 Predict Churn Probability"):

    # 1. Construct the complete input dictionary required by the model.
    #    Note: The column names here must exactly match the ones used for training!
    #    For features we didn't create input widgets for, we use reasonable default values.
    inference_data = {
        "customerID": "webapp-test-user", # <-- Temporary fix for the old model
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # 2. Wrap the payload in the format required by SageMaker.
    payload = {
        "instances": [inference_data]
    }

    # 3. Invoke the SageMaker endpoint
    try:
        with st.spinner('Model is predicting...'):
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
        
        result = json.loads(response["Body"].read().decode())
        prediction_prob = result['predictions'][0]

        # 4. Display the results on the main interface
        st.subheader("📊 Prediction Result")
        
        # Use a progress bar to visualize the probability
        st.progress(prediction_prob)
        st.success(f"The model predicts a churn probability of: **{prediction_prob:.2%}**")

        if prediction_prob > 0.5:
            st.warning("🚨 High Churn Risk! Customer retention measures are recommended.")
        else:
            st.balloons()
            st.info("✅ Low Churn Risk. The customer is stable and can be considered for upselling opportunities.")
            
        # (Optional) Display the raw data sent to the model
        with st.expander("View the detailed data sent to the model"):
            st.json(payload)

    except Exception as e:
        st.error(f"An error occurred while calling the endpoint: {e}")