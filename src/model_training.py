import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from termcolor import colored
import os
from datetime import datetime

def load_processed_data():
    """load processed data from csv files"""
    print(colored('Loading processed data...', 'blue'))
    
    try:
        #load features and target
        X = pd.read_csv('data/processed/X_train.csv')
        y = pd.read_csv('data/processed/y_train.csv')
        
        print(colored('Data loaded successfully', 'green'))
        return X, y
    except Exception as e:
        print(colored(f'Error loading data: {str(e)}', 'red'))
        raise

def train_baseline_models(X, y):
    """train baseline models and select the best one"""
    print(colored('Training baseline models...', 'blue'))
    
    try:
        #initialize models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        #train and evaluate each model
        results = {}
        for name, model in models.items():
            print(colored(f'\nTraining {name}...', 'blue'))
            model.fit(X, y.values.ravel())
            
            #make predictions
            y_pred = model.predict(X)
            
            #calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(colored(f'{name} performance:', 'green'))
            print(f'Accuracy: {accuracy:.3f}')
            print(f'Precision: {precision:.3f}')
            print(f'Recall: {recall:.3f}')
            print(f'F1 Score: {f1:.3f}')
        
        #select best model based on f1 score
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']
        
        print(colored(f'\nBest model: {best_model_name}', 'green'))
        return best_model, results
    except Exception as e:
        print(colored(f'Error training models: {str(e)}', 'red'))
        raise

def optimize_hyperparameters(model, X, y):
    """optimize hyperparameters using grid search"""
    print(colored('Optimizing hyperparameters...', 'blue'))
    
    try:
        if isinstance(model, RandomForestClassifier):
            #define parameter grid for random forest
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            #define parameter grid for logistic regression
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        
        #perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X, y.values.ravel())
        
        print(colored('Best parameters:', 'green'))
        print(grid_search.best_params_)
        
        return grid_search.best_estimator_
    except Exception as e:
        print(colored(f'Error optimizing hyperparameters: {str(e)}', 'red'))
        raise

def save_model(model, results):
    """save the trained model and results"""
    print(colored('Saving model and results...', 'blue'))
    
    try:
        #create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        #save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/random_forest_model_{timestamp}.joblib'
        joblib.dump(model, model_path)
        
        #prepare serializable results
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'accuracy': float(model_results['accuracy']),
                'precision': float(model_results['precision']),
                'recall': float(model_results['recall']),
                'f1': float(model_results['f1'])
            }
        
        #save results
        results_path = f'models/training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(colored('Model and results saved successfully', 'green'))
        return model_path, results_path
    except Exception as e:
        print(colored(f'Error saving model and results: {str(e)}', 'red'))
        raise

def main():
    """main function to run the training pipeline"""
    print(colored('Starting model training pipeline...', 'blue'))
    
    try:
        #load data
        X, y = load_processed_data()
        
        #train baseline models
        best_model, results = train_baseline_models(X, y)
        
        #optimize hyperparameters
        optimized_model = optimize_hyperparameters(best_model, X, y)
        
        #save model and results
        model_path, results_path = save_model(optimized_model, results)
        
        print(colored('\nTraining pipeline completed successfully!', 'green'))
        print(colored(f'Model saved to: {model_path}', 'blue'))
        print(colored(f'Results saved to: {results_path}', 'blue'))
    except Exception as e:
        print(colored(f'\nTraining pipeline failed: {str(e)}', 'red'))
        raise

if __name__ == '__main__':
    main() 