# Predictive Maintenance API

This API provides real-time predictions for equipment failures based on sensor data. It uses a trained machine learning model to predict the probability of failure for individual products or batches of products.

## Features

- Single product prediction
- Batch prediction for multiple products
- Health check endpoint
- Input validation
- Error handling
- Detailed logging

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the model and scaler files are in the correct locations:
- `models/random_forest_model.joblib`
- `data/processed/scaler.pkl`

## Running the API

Start the API server:
```bash
python src/api/main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
```
GET /health
```
Returns the health status of the API and model.

### 2. Single Prediction
```
POST /predict
```
Predicts failure probability for a single product.

Request body:
```json
{
    "product_id": "M14860",
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0
}
```

### 3. Batch Prediction
```
POST /predict/batch
```
Predicts failure probability for multiple products.

Request body:
```json
{
    "data": [
        {
            "product_id": "M14860",
            "type": "M",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0
        },
        {
            "product_id": "L47181",
            "type": "L",
            "air_temperature": 298.2,
            "process_temperature": 308.7,
            "rotational_speed": 1408,
            "torque": 46.3,
            "tool_wear": 3
        }
    ]
}
```

## Testing

Run the test script to verify API functionality:
```bash
python src/api/test_api.py
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The API includes comprehensive error handling for:
- Invalid input data
- Missing required fields
- Model loading errors
- Processing errors

## Logging

The API logs all requests and errors to the console with colored output for better visibility.

## Example Usage

### Python
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "product_id": "M14860",
        "type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "data": [
            {
                "product_id": "M14860",
                "type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0
            }
        ]
    }
)
print(response.json())
```

### cURL
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "product_id": "M14860",
         "type": "M",
         "air_temperature": 298.1,
         "process_temperature": 308.6,
         "rotational_speed": 1551,
         "torque": 42.8,
         "tool_wear": 0
     }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
         "data": [
             {
                 "product_id": "M14860",
                 "type": "M",
                 "air_temperature": 298.1,
                 "process_temperature": 308.6,
                 "rotational_speed": 1551,
                 "torque": 42.8,
                 "tool_wear": 0
             }
         ]
     }'
``` 