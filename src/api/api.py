import os
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from termcolor import colored
import joblib

#initialize fastapi app
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures from CSV data",
    version="1.0.0"
)

#load model and scaler
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "random_forest_model.joblib"
SCALER_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(colored("Model and scaler loaded successfully", "green"))
except Exception as e:
    print(colored(f"Error loading model or scaler: {str(e)}", "red"))
    raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """preprocess input data for model prediction"""
    try:
        #create base features
        features = pd.DataFrame({
            'Type': pd.Categorical(df['Type']).codes,  #convert type to numeric
            'Air_temperature___C_': df['Air temperature [°C]'],
            'Process_temperature___C_': df['Process temperature [°C]'],
            'Rotational_speed__rpm_': df['Rotational speed [rpm]'],
            'Torque__Nm_': df['Torque [Nm]'],
            'Tool_wear__min_': df['Tool wear [min]']
        })
        
        #add engineered features
        features['Temperature_Difference'] = features['Process_temperature___C_'] - features['Air_temperature___C_']
        features['Power'] = features['Torque__Nm_'] * features['Rotational_speed__rpm_']
        features['Tool_Wear_Rate'] = features['Tool_wear__min_'] / features['Rotational_speed__rpm_']
        features['Temp_Tool_Interaction'] = features['Process_temperature___C_'] * features['Tool_wear__min_']
        features['Speed_Torque_Interaction'] = features['Rotational_speed__rpm_'] * features['Torque__Nm_']
        
        #scale features
        scaled_data = scaler.transform(features)
        return pd.DataFrame(scaled_data, columns=features.columns)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")

@app.get("/health")
def health_check():
    """check api health status"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """predict failure probabilities from csv"""
    try:
        #read csv file
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        #store product ids
        product_ids = df['Product ID'].tolist()
        
        #preprocess data
        processed_data = preprocess_data(df)
        
        #make predictions
        probabilities = model.predict_proba(processed_data)[:, 1]  #get failure probability
        
        #prepare results
        results = []
        for i, product_id in enumerate(product_ids):
            results.append({
                "product_id": product_id,
                "failure_probability": float(probabilities[i])
            })
        
        return {
            "status": "success",
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 