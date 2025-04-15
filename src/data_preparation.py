import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from termcolor import colored
import os
import re
import joblib

def clean_feature_name(name):
    """clean feature name for xgboost compatibility"""
    #replace special chars with underscores
    cleaned = re.sub(r'[\[\]<>°\s]', '_', name)
    #remove non-alphanumeric chars
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', cleaned)
    return cleaned

def load_raw_data():
    """load raw data from csv"""
    print(colored('Loading raw data...', 'blue'))
    data = pd.read_csv('data/raw/predictive_maintenance.csv')
    print(colored(f'Loaded {len(data)} records with {len(data.columns)} features', 'green'))
    return data

def preprocess_data(data):
    """preprocess and encode data"""
    print(colored('Preprocessing data...', 'blue'))
    
    #drop unnecessary columns
    X = data.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    
    #clean feature names
    X.columns = [clean_feature_name(col) for col in X.columns]
    
    #encode categorical variables
    le = LabelEncoder()
    X['Type'] = le.fit_transform(X['Type'])
    
    #get target variable
    y = data['Target']
    
    print(colored('Data preprocessing completed', 'green'))
    return X, y

def engineer_features(X):
    """create new features"""
    print(colored('Engineering new features...', 'blue'))
    
    #create temperature difference
    X['Temperature_Difference'] = X['Process_temperature___C_'] - X['Air_temperature___C_']
    
    #create power feature
    X['Power'] = X['Torque__Nm_'] * X['Rotational_speed__rpm_']
    
    #create tool wear rate
    X['Tool_Wear_Rate'] = X['Tool_wear__min_'] / X['Rotational_speed__rpm_']
    
    #create interaction features
    X['Temp_Tool_Interaction'] = X['Process_temperature___C_'] * X['Tool_wear__min_']
    X['Speed_Torque_Interaction'] = X['Rotational_speed__rpm_'] * X['Torque__Nm_']
    
    print(colored('Feature engineering completed', 'green'))
    return X

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """split data and scale features"""
    print(colored('Splitting and scaling data...', 'blue'))
    
    #split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    #print class distribution
    print(colored('\nClass Distribution:', 'blue'))
    print('Training set:')
    print(y_train.value_counts(normalize=True))
    print('\nTest set:')
    print(y_test.value_counts(normalize=True))
    
    #scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #convert to dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(colored('Data splitting and scaling completed', 'green'))
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """save processed data and scaler"""
    print(colored('Saving processed data...', 'blue'))
    
    #ensure consistent feature order
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
    
    X_train = X_train[feature_order]
    X_test = X_test[feature_order]
    
    #save as numpy arrays
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    #save as csv files
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    #save scaler
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    print(colored('Processed data saved successfully', 'green'))
    print(colored('Files saved in data/processed:', 'blue'))
    print('- X_train.npy and X_train.csv: Training features')
    print('- X_test.npy and X_test.csv: Testing features')
    print('- y_train.npy and y_train.csv: Training labels')
    print('- y_test.npy and y_test.csv: Testing labels')
    print('- scaler.pkl: Feature scaler')

def main():
    """run data preparation pipeline"""
    print(colored('Starting data preparation pipeline...', 'blue'))
    
    #create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    #load and preprocess data
    df = load_raw_data()
    X, y = preprocess_data(df)
    
    #engineer features
    X = engineer_features(X)
    
    #split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    #save processed data
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    
    print(colored('\nData preparation completed successfully!', 'green'))
    print(colored('Processed data saved in data/processed directory', 'blue'))

if __name__ == "__main__":
    main() 