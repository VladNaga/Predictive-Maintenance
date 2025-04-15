# Predictive Maintenance System

A machine learning-based system for predicting equipment failures in manufacturing environments. The system processes sensor data, trains predictive models, and provides real-time failure predictions through a REST API.

## Project Structure

```
.
├── data/                    # Data directory
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── models/                 # Trained models
├── src/                    # Source code
│   ├── api/               # API implementation
│   │   ├── api.py        # FastAPI application
│   │   └── test_api.py   # API test script
│   ├── main.py           # Main pipeline for insights and results
│   ├── data_preparation.py # Data preprocessing
│   ├── model_training.py  # Model training pipeline
│   ├── model_analysis.py  # Model evaluation and visualization
│   ├── business_analysis.py # Business impact analysis
│   └── generate_test_data.py # Test data generation utility
└── requirements.txt        # Project dependencies
```

## Components and Workflow

### 1. Data Preparation (`src/data_preparation.py`)
- Loads raw sensor data
- Performs data cleaning and preprocessing
- Engineers new features
- Splits data into training and test sets
- Saves processed data for model training

### 2. Model Training (`src/model_training.py`)
- Loads processed training data
- Trains multiple baseline models (Random Forest, Logistic Regression)
- Evaluates model performance
- Optimizes hyperparameters
- Saves the best model and training results

### 3. Model Analysis (`src/model_analysis.py`)
- Evaluates model performance on test data
- Generates performance metrics and visualizations
- Creates confusion matrix and ROC curves
- Analyzes feature importance
- Generates comprehensive documentation

### 4. Business Analysis (`src/business_analysis.py`)
- Calculates business value and impact metrics
- Generates system architecture documentation
- Creates presentation materials
- Analyzes cost savings and reliability metrics

### 5. API Implementation (`src/api/api.py`)
- Provides REST API endpoints for predictions
- Handles data preprocessing and validation
- Returns failure probabilities and predictions
- Includes health check and monitoring

### 6. Main Pipeline (`src/main.py`)
- Orchestrates the entire workflow
- Runs data preparation, model training, and analysis
- Generates comprehensive insights and results
- Produces business impact documentation
- Handles error logging and reporting

### 7. Test Data Generation (`src/generate_test_data.py`)
- Creates realistic test datasets for API validation
- Maintains data distributions and patterns from original data
- Ensures realistic failure scenarios
- Generates balanced machine types (L, M, H)
- Produces consistent sensor readings and tool wear patterns

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main pipeline to generate all insights and results:
```bash
python src/main.py
```

This will:
- Prepare and process the data
- Train and evaluate the model
- Generate analysis and visualizations
- Create business impact documentation
- Test the API with generated data

3. Generate test data (optional):
```bash
python src/generate_test_data.py
```

4. Start the API:
```bash
python src/api/api.py
```

5. Test the API:
```bash
python src/api/test_api.py
```

## API Usage

The API provides the following endpoints:

### Health Check
```bash
GET /health
```
Returns the health status of the API and model.

### CSV Prediction
```bash
POST /predict/csv
```
Accepts a CSV file with machine data and returns failure probabilities.

Example request:
```bash
curl -X POST "http://localhost:8000/predict/csv" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/raw/predictive_maintenance_test.csv"
```

## Data Format

The input CSV file should contain the following columns:
- `Product ID`: Unique identifier for each machine
- `Type`: Machine type (L, M, H)
- `Air temperature [°C]`: Air temperature in Celsius
- `Process temperature [°C]`: Process temperature in Celsius
- `Rotational speed [rpm]`: Rotational speed in RPM
- `Torque [Nm]`: Torque in Newton-meters
- `Tool wear [min]`: Tool wear in minutes

