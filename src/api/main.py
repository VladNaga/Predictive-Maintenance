from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import os
from datetime import datetime
import logging
from termcolor import colored

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#initialize fastapi app
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting equipment failures based on sensor data",
    version="1.0.0"
)

#load model and scaler
try:
    model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('data/processed/scaler.pkl')
    logger.info(colored("Model and scaler loaded successfully", "green"))
except Exception as e:
    logger.error(colored(f"Error loading model or scaler: {str(e)}", "red"))
    raise

#define request/response models
class SensorData(BaseModel):
    product_id: str
    type: str
    air_temperature___C_: float = Field(..., alias="air_temperature")
    process_temperature___C_: float = Field(..., alias="process_temperature")
    rotational_speed__rpm_: float = Field(..., alias="rotational_speed")
    torque__Nm_: float = Field(..., alias="torque")
    tool_wear__min_: float = Field(..., alias="tool_wear")
    
    class Config:
        allow_population_by_field_name = True
        populate_by_name = True

class PredictionResponse(BaseModel):
    product_id: str
    failure_probability: float
    prediction: str
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    data: List[SensorData]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

#helper functions
def preprocess_data(data: SensorData) -> pd.DataFrame:
    """preprocess input data for prediction"""
    try:
        #create dataframe with correct feature names
        input_data = {
            'Type': data.type,
            'Air_temperature___C_': data.air_temperature___C_,
            'Process_temperature___C_': data.process_temperature___C_,
            'Rotational_speed__rpm_': data.rotational_speed__rpm_,
            'Torque__Nm_': data.torque__Nm_,
            'Tool_wear__min_': data.tool_wear__min_
        }
        
        #create dataframe
        df = pd.DataFrame([input_data])
        
        #add engineered features
        df['Temperature_Difference'] = df['Process_temperature___C_'] - df['Air_temperature___C_']
        df['Power'] = df['Torque__Nm_'] * df['Rotational_speed__rpm_']
        df['Tool_Wear_Rate'] = df['Tool_wear__min_'] / df['Rotational_speed__rpm_']
        df['Temp_Tool_Interaction'] = df['Process_temperature___C_'] * df['Tool_wear__min_']
        df['Speed_Torque_Interaction'] = df['Rotational_speed__rpm_'] * df['Torque__Nm_']
        
        #ensure correct feature order
        feature_order = [
            'Type',
            'Air_temperature___C_',
            'Process_temperature___C_',
            'Rotational_speed__rpm_',
            'Torque__Nm_',
            'Tool_wear__min_',
            'Temperature_Difference',
            'Power',
            'Tool_Wear_Rate',
            'Temp_Tool_Interaction',
            'Speed_Torque_Interaction'
        ]
        
        #reorder columns
        df = df[feature_order]
        
        #scale features
        scaled_data = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_order)
        
        return scaled_df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(f"Input data: {input_data}")
        logger.error(f"DataFrame columns: {df.columns.tolist() if 'df' in locals() else 'DataFrame not created'}")
        raise HTTPException(status_code=500, detail=str(e))

def get_prediction(data: SensorData) -> PredictionResponse:
    """get prediction for single data point"""
    try:
        #preprocess data
        processed_data = preprocess_data(data)
        
        #get prediction probability
        proba = model.predict_proba(processed_data)[0][1]
        
        #determine prediction
        prediction = "Failure" if proba > 0.5 else "Normal"
        
        return PredictionResponse(
            product_id=data.product_id,
            failure_probability=float(proba),
            prediction=prediction,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(colored(f"Error in prediction: {str(e)}", "red"))
        raise HTTPException(status_code=500, detail=str(e))

#api endpoints
@app.get("/")
async def root():
    """root endpoint"""
    return {
        "message": "Predictive Maintenance API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData):
    """predict failure probability for a single product"""
    logger.info(colored(f"Received prediction request for product {data.product_id}", "blue"))
    return get_prediction(data)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """predict failure probability for multiple products"""
    logger.info(colored(f"Received batch prediction request for {len(request.data)} products", "blue"))
    predictions = [get_prediction(item) for item in request.data]
    return BatchPredictionResponse(predictions=predictions)

@app.get("/health")
async def health_check():
    """health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 