# Debug script to check model's feature names
import joblib
import pandas as pd
from termcolor import colored

# Load model and scaler
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('data/processed/scaler.pkl')

# Print model information
print(colored("\nModel Feature Names:", "blue"))
print(model.feature_names_in_.tolist())

# Load a sample of training data to check feature names
X_train = pd.read_csv('data/processed/X_train.csv')
print(colored("\nTraining Data Columns:", "blue"))
print(X_train.columns.tolist()) 