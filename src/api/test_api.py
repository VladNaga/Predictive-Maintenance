import requests
from termcolor import colored
import time
import os
from pathlib import Path

#api configuration
BASE_URL = "http://localhost:8001"
TEST_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "predictive_maintenance_test.csv"

def test_health_check():
    """test health check endpoint"""
    print(colored("\nTesting health check endpoint...", "blue"))
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(colored("✓ Health check passed", "green"))
            print(response.json())
        else:
            print(colored("✗ Health check failed", "red"))
            print(response.json())
    except Exception as e:
        print(colored(f"✗ Health check error: {str(e)}", "red"))

def test_csv_prediction():
    """test csv prediction endpoint"""
    print(colored("\nTesting CSV prediction endpoint...", "blue"))
    
    try:
        #check if test csv exists
        if not TEST_CSV_PATH.exists():
            print(colored(f"✗ Test CSV not found at {TEST_CSV_PATH}", "red"))
            return
            
        #send csv for prediction
        with open(TEST_CSV_PATH, 'rb') as f:
            files = {'file': ('predictive_maintenance_test.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/predict/csv", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(colored("✓ CSV prediction successful", "green"))
            print("\nSample predictions (first 5):")
            for pred in result['predictions'][:5]:
                print(f"Product ID: {pred['product_id']}, Failure Probability: {pred['failure_probability']:.2%}")
            
            #print risk distribution
            probabilities = [pred['failure_probability'] for pred in result['predictions']]
            high_risk = sum(1 for p in probabilities if p > 0.7)
            medium_risk = sum(1 for p in probabilities if 0.3 <= p <= 0.7)
            low_risk = sum(1 for p in probabilities if p < 0.3)
            
            print("\nRisk Distribution:")
            print(f"High Risk (>70%): {high_risk} machines")
            print(f"Medium Risk (30-70%): {medium_risk} machines")
            print(f"Low Risk (<30%): {low_risk} machines")
            
        else:
            print(colored("✗ CSV prediction failed", "red"))
            print(response.json())
            
    except Exception as e:
        print(colored(f"✗ Error during CSV prediction: {str(e)}", "red"))

def main():
    """run all tests"""
    print(colored("Starting API tests...", "blue"))
    
    #wait for api to start
    print(colored("Waiting for API to start...", "blue"))
    time.sleep(2)
    
    #run tests
    test_health_check()
    test_csv_prediction()
    
    print(colored("\nAll tests completed!", "green"))

if __name__ == "__main__":
    main() 