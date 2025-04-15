import pandas as pd
import numpy as np
from termcolor import colored
import os

def generate_test_data(num_samples=100):
    """generate test dataset with realistic machine data"""
    print(colored("Generating test dataset...", "blue"))
    
    #initialize data columns
    data = {
        'UDI': [],
        'Product ID': [],
        'Type': [],
        'Air temperature [°C]': [],
        'Process temperature [°C]': [],
        'Rotational speed [rpm]': [],
        'Torque [Nm]': [],
        'Tool wear [min]': [],
        'Target': [],
        'Failure Type': []
    }
    
    #generate data for each sample
    for i in range(1, num_samples + 1):
        #generate machine type
        machine_type = np.random.choice(['L', 'M', 'H'], p=[0.5, 0.3, 0.2])
        
        #generate base temperatures
        if machine_type == 'L':
            air_temp = np.random.normal(25.0, 0.2)
            process_temp = np.random.normal(35.0, 0.2)
            speed = np.random.normal(1500, 100)
            torque = np.random.normal(40, 5)
        elif machine_type == 'M':
            air_temp = np.random.normal(25.2, 0.2)
            process_temp = np.random.normal(35.2, 0.2)
            speed = np.random.normal(1700, 100)
            torque = np.random.normal(35, 5)
        else:  #H
            air_temp = np.random.normal(25.4, 0.2)
            process_temp = np.random.normal(35.4, 0.2)
            speed = np.random.normal(1900, 100)
            torque = np.random.normal(30, 5)
        
        #ensure speed bounds
        speed = max(1200, min(2500, speed))
        
        #generate tool wear
        tool_wear = min(200, max(0, int(i * 2 + np.random.normal(0, 3))))
        
        #determine failure
        failure = 0
        failure_type = "No Failure"
        
        #generate failure conditions
        if tool_wear > 180:  #tool wear failure
            failure = 1
            failure_type = "Tool Wear Failure"
        elif speed > 2400:  #power failure
            failure = 1
            failure_type = "Power Failure"
        elif torque > 55:  #overstrain failure
            failure = 1
            failure_type = "Overstrain Failure"
        
        #add data
        data['UDI'].append(i)
        data['Product ID'].append(f"TEST{i:05d}")
        data['Type'].append(machine_type)
        data['Air temperature [°C]'].append(round(air_temp, 2))
        data['Process temperature [°C]'].append(round(process_temp, 2))
        data['Rotational speed [rpm]'].append(round(speed))
        data['Torque [Nm]'].append(round(torque, 1))
        data['Tool wear [min]'].append(tool_wear)
        data['Target'].append(failure)
        data['Failure Type'].append(failure_type)
    
    #create dataframe
    df = pd.DataFrame(data)
    
    #save to csv
    output_path = 'data/raw/predictive_maintenance_test.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(colored(f"Test dataset generated with {num_samples} samples", "green"))
    print(colored(f"Dataset saved to {output_path}", "green"))
    
    #print summary
    print("\nDataset Summary:")
    print(f"Number of samples: {len(df)}")
    print(f"Number of failures: {df['Target'].sum()}")
    print("\nFailure Types:")
    print(df['Failure Type'].value_counts())
    print("\nMachine Types:")
    print(df['Type'].value_counts())

if __name__ == "__main__":
    generate_test_data(100) 