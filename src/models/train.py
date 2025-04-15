import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
from termcolor import colored

def engineer_features(df):
    """Engineer new features from existing ones."""
    df_engineered = df.copy()
    
    # 1. Temperature difference (process vs air)
    df_engineered['Temperature_Difference'] = df['Process temperature [°C]'] - df['Air temperature [°C]']
    
    # 2. Power estimation (Torque × Rotational speed)
    df_engineered['Power_Estimate'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    
    # 3. Wear rate (Tool wear / UDI as a proxy for time)
    df_engineered['Wear_Rate'] = df['Tool wear [min]'] / df['UDI']
    
    # 4. Operational Stress (Power × Wear)
    df_engineered['Operational_Stress'] = df_engineered['Power_Estimate'] * df['Tool wear [min]']
    
    # 5. Temperature Stress (Temperature difference × Rotational speed)
    df_engineered['Temperature_Stress'] = df_engineered['Temperature_Difference'] * df['Rotational speed [rpm]']
    
    return df_engineered

def prepare_data(df):
    """Prepare data for model training."""
    # Select features
    feature_cols = ['Air temperature [°C]', 'Process temperature [°C]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                   'Temperature_Difference', 'Power_Estimate', 'Wear_Rate',
                   'Operational_Stress', 'Temperature_Stress', 'Type']
    
    X = df[feature_cols].copy()
    y = df['Target']
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['Type'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model():
    """Train and select the best model."""
    print(colored('Loading data...', 'blue'))
    df = pd.read_csv('data/predictive_maintenance.csv')
    
    print(colored('Engineering features...', 'blue'))
    df_engineered = engineer_features(df)
    
    print(colored('Preparing data...', 'blue'))
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df_engineered)
    
    # Define models to test
    models = {
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    # Train and evaluate models
    best_score = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(colored(f'\nTraining {name}...', 'blue'))
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate F1 score for the minority class (failures)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_score = report['1']['f1-score']
        
        print(colored(f'{name} Performance:', 'green'))
        print(classification_report(y_test, y_pred))
        
        # Update best model if current one is better
        if f1_score > best_score:
            best_score = f1_score
            best_model = model
            best_model_name = name
    
    print(colored(f'\nBest model: {best_model_name} (F1-score: {best_score:.3f})', 'green'))
    
    # Save best model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/predictive_maintenance_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print(colored('\nBest model and scaler saved successfully!', 'green'))
    
    return best_model, scaler

if __name__ == "__main__":
    train_model() 