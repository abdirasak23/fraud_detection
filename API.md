# Credit Card Fraud Detection API Documentation

## Overview

The Credit Card Fraud Detection API provides real-time fraud detection capabilities for credit card transactions. Built using Flask, it leverages a trained XGBoost model to predict fraudulent transactions and provides explainable results using SHAP values.

## Base URL

```
http://localhost:5000
```

## Endpoints

### 1. Health Check

Check if the API is running.

```
GET /
```

**Response**
```
🚀 XGBoost Fraud Detection API is Running!
```

### 2. Predict Fraud

Make a fraud prediction for a credit card transaction.

```
POST /predict
```

#### Request Body

```json
{
    "features": [
        0.0,    // V1
        0.0,    // V2
        0.0,    // V3
        ...     // Up to V30 (30 values total)
    ]
}
```

#### Response

```json
{
    "fraud_probability": 0.123456,
    "prediction": "Normal",
    "shap_values": [
        0.1,    // Impact of V1
        -0.2,   // Impact of V2
        0.3,    // Impact of V3
        ...     // Up to V30
    ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| fraud_probability | float | Probability of the transaction being fraudulent (0-1) |
| prediction | string | "Fraud" or "Normal" based on threshold |
| shap_values | array | SHAP values explaining the model's decision |

#### Error Responses

1. Missing Features
```json
{
    "error": "Missing 'features' key in JSON input"
}
```

2. Invalid Feature Count
```json
{
    "error": "Expected 30 features, got X"
}
```

## Implementation Details

### Model Loading

The API loads a pre-trained XGBoost model on startup:

```python
model = joblib.load("xgboost_fraud_model.pkl")
```

### SHAP Integration

SHAP values are calculated for model explainability:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer(features)
```

### Visual Alerts

The API includes OpenCV integration for visual fraud alerts:

```python
def show_fraud_alert():
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(img, "⚠️ FRAUD DETECTED!", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Fraud Alert", img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
```

## Security Considerations

1. Input Validation
   - Verify feature count (must be 30)
   - Validate data types (must be numeric)
   - Check value ranges

2. Rate Limiting
   - Consider implementing rate limiting for production use
   - Prevent DoS attacks

3. Error Handling
   - Graceful handling of invalid inputs
   - Clear error messages
   - No stack traces in production

## Testing

### Sample cURL Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'
```

### Sample Python Request

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "features": [0] * 30  # Array of 30 features
}

response = requests.post(url, json=data)
print(response.json())
```

## Performance Considerations

1. Model Loading
   - Model is loaded once at startup
   - SHAP explainer is initialized once

2. Concurrent Processing
   - API runs with threading enabled
   - Can handle multiple simultaneous requests

3. Memory Management
   - Efficient numpy array handling
   - No memory leaks in OpenCV display

## Logging

The API logs predictions and alerts to `fraud_api.log`:

```python
print(f"🚨 Fraud detected! Probability: {fraud_probability:.4f}")
print(f"✅ Normal transaction. Probability: {fraud_probability:.4f}")
```

## Future Improvements

1. Authentication
   - Add API key authentication
   - Implement JWT tokens

2. Monitoring
   - Add request/response logging
   - Implement performance metrics

3. Caching
   - Cache frequent predictions
   - Implement result caching

4. Documentation
   - Add Swagger/OpenAPI documentation
   - Include more example requests
