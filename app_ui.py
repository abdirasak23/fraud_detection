import streamlit as st
import requests
import numpy as np
import shap
import matplotlib.pyplot as plt

# Streamlit UI Configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection with Explainability")

# Sidebar for user input
st.sidebar.header("Enter Transaction Features")
features = [st.sidebar.number_input(f"Feature {i+1}", value=0.0) for i in range(30)]

# Prediction Button
if st.sidebar.button("🔍 Predict Fraud"):
    api_url = "http://127.0.0.1:5000/predict"
    response = requests.post(api_url, json={"features": features})

    if response.status_code == 200:
        data = response.json()

        # 🔍 Debugging: Print API Response
        print("API Response:", data)

        # Extract fraud probability and prediction
        fraud_probability = data.get("fraud_probability", None)
        prediction_label = "⚠️ Fraud" if data.get("prediction", 0) == 1 else "✅ Normal"

        # Display prediction results
        st.markdown("### 📊 Prediction Result")
        st.metric(label="Fraud Probability", value=f"{fraud_probability:.6f}" if fraud_probability is not None else "N/A")
        st.metric(label="Prediction", value=prediction_label)

        # 🔎 Handle SHAP values
        if "shap_values" in data and data["shap_values"] is not None:
            shap_values = np.array(data["shap_values"]).reshape(-1)
            feature_names = [f"Feature {i+1}" for i in range(30)]

            # SHAP Summary Plot
            st.markdown("### 🔎 SHAP Explainability")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.bar_plot(shap_values, feature_names, show=False)
            st.pyplot(fig)

            # Key Insights from SHAP
            sorted_features = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
            st.markdown("### 📝 Key Insights from SHAP")
            for feature, impact in sorted_features:
                color = "🟢" if impact < 0 else "🔴"
                direction = "Decreased" if impact < 0 else "Increased"
                st.markdown(f"- **{feature}** {color} {direction} fraud likelihood.")

        else:
            st.error("❌ SHAP values missing from API response.")
            st.json(data)  # Show raw API response

    else:
        st.error("❌ API request failed. Please check if Flask is running.")
