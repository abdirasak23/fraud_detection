from flask import Flask, request, jsonify
import joblib
import numpy as np
import shap

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("xgboost_fraud_model.pkl")

# Use TreeExplainer for XGBoost models
explainer = shap.TreeExplainer(model)

# ✅ Home Route
@app.route('/')
def home():
    return "🚀 XGBoost Fraud Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in JSON input"}), 400

    features = np.array(data["features"]).reshape(1, -1)

    if features.shape[1] != 30:
        return jsonify({"error": f"Expected 30 features, got {features.shape[1]}"}), 400

    # Predict fraud probability
    fraud_probability = model.predict_proba(features)[0][1]

    # ✅ Set Fraud Threshold to 0.5
    fraud_threshold = 0.5  
    prediction = 1 if fraud_probability > fraud_threshold else 0  

    # Calculate SHAP values
    shap_values = explainer.shap_values(features)

    response = {
        "fraud_probability": round(float(fraud_probability), 6),
        "prediction": prediction,
        "shap_values": shap_values.tolist()  # Convert to JSON serializable format
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
