import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
import os
import json
from datetime import datetime

def generate_system_architecture():
    """generate system architecture documentation and diagram"""
    print(colored('Generating system architecture documentation...', 'blue'))
    
    #create architecture directory
    os.makedirs('documentation/architecture', exist_ok=True)
    
    #system architecture description
    architecture = {
        "components": {
            "data_collection": {
                "description": "Real-time sensor data collection from manufacturing equipment",
                "technologies": ["IoT sensors", "SCADA systems", "Data acquisition systems"]
            },
            "data_processing": {
                "description": "Data preprocessing and feature engineering pipeline",
                "technologies": ["Apache Spark", "Python", "Pandas"]
            },
            "model_serving": {
                "description": "Model deployment and prediction service",
                "technologies": ["FastAPI", "Docker", "Kubernetes"]
            },
            "alerting_system": {
                "description": "Real-time alert generation and notification system",
                "technologies": ["Kafka", "Slack API", "Email notifications"]
            },
            "dashboard": {
                "description": "Business intelligence and monitoring dashboard",
                "technologies": ["Grafana", "Tableau", "Power BI"]
            }
        },
        "integration_points": {
            "erp_system": "Integration with existing ERP for maintenance scheduling",
            "crm_system": "Customer notification and service planning",
            "maintenance_system": "Automated maintenance work order generation"
        }
    }
    
    #save architecture documentation
    with open('documentation/architecture/system_architecture.json', 'w') as f:
        json.dump(architecture, f, indent=4)
    
    print(colored('System architecture documentation generated', 'green'))

def calculate_business_value():
    """calculate theoretical business value and impact metrics"""
    print(colored('Calculating theoretical business value...', 'blue'))
    
    #theoretical performance metrics (based on industry standards)
    theoretical_metrics = {
        "accuracy": 0.95,
        "precision": 0.90,
        "recall": 0.85,
        "f1_score": 0.87
    }
    
    #theoretical class distribution
    theoretical_dist = {
        "class_0": 900,  #non-failure cases
        "class_1": 100   #failure cases
    }
    
    #theoretical business assumptions (clearly marked as estimates)
    business_assumptions = {
        "avg_maintenance_cost": 5000,    #average cost of unplanned maintenance
        "avg_downtime_cost": 1000,       #cost per hour of downtime
        "avg_repair_time": 8,            #average hours to repair
        "annual_equipment_count": 1000,   #number of equipment units
        "early_detection_savings_rate": 0.3,  #30% cost reduction from early detection
        "note": "These are theoretical values for demonstration purposes only"
    }
    
    #calculate theoretical cost savings
    true_positives = int(theoretical_metrics['recall'] * theoretical_dist['class_1'])
    false_negatives = theoretical_dist['class_1'] - true_positives
    
    early_detection_savings = true_positives * business_assumptions['avg_maintenance_cost'] * business_assumptions['early_detection_savings_rate']
    downtime_savings = (false_negatives + true_positives) * business_assumptions['avg_downtime_cost'] * business_assumptions['avg_repair_time']
    total_savings = early_detection_savings + downtime_savings
    
    business_value = {
        "disclaimer": "This is a theoretical analysis based on industry standards and assumptions. Actual values may vary significantly.",
        "cost_savings": {
            "early_detection": early_detection_savings,
            "downtime_reduction": downtime_savings,
            "total_annual_savings": total_savings,
            "assumptions": business_assumptions
        },
        "reliability_metrics": {
            "prediction_accuracy": theoretical_metrics['accuracy'],
            "failure_detection_rate": theoretical_metrics['recall'],
            "false_alarm_rate": 1 - theoretical_metrics['precision']
        },
        "customer_impact": {
            "reduced_downtime_hours": (false_negatives + true_positives) * business_assumptions['avg_repair_time'],
            "improved_service_quality": theoretical_metrics['precision'] * 100,
            "proactive_maintenance_rate": true_positives / (true_positives + false_negatives)
        },
        "model_performance": {
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "total_samples": theoretical_dist['class_0'] + theoretical_dist['class_1']
        }
    }
    
    #save business value analysis
    with open('documentation/business_value.json', 'w') as f:
        json.dump(business_value, f, indent=4)
    
    print(colored('Theoretical business value analysis generated', 'green'))
    return business_value

def generate_presentation_materials(business_value, architecture):
    """generate presentation materials including slides and executive summary"""
    print(colored('Generating presentation materials...', 'blue'))
    
    #create presentation directory
    os.makedirs('documentation/presentation', exist_ok=True)
    
    #generate executive summary
    executive_summary = f"""# Predictive Maintenance System - Executive Summary

## Note
This is a theoretical analysis based on industry standards and assumptions. Actual values may vary significantly.

## Business Impact
- Theoretical Annual Cost Savings: €{business_value['cost_savings']['total_annual_savings']:,.2f}
- Theoretical Reduced Downtime: {business_value['customer_impact']['reduced_downtime_hours']:,.0f} hours
- Theoretical Service Quality Improvement: {business_value['customer_impact']['improved_service_quality']:.1f}%

## System Overview
The predictive maintenance system integrates real-time sensor data with advanced machine learning to:
1. Predict equipment failures before they occur
2. Optimize maintenance schedules
3. Reduce operational costs
4. Improve customer satisfaction

## Key Components
- Real-time data collection and processing
- Machine learning model for failure prediction
- Automated alerting system
- Integration with existing enterprise systems

## Next Steps
1. Pilot implementation with selected equipment
2. System integration and testing
3. Full-scale deployment
4. Continuous monitoring and improvement

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    #save executive summary
    with open('documentation/presentation/executive_summary.md', 'w') as f:
        f.write(executive_summary)
    
    print(colored('Presentation materials generated', 'green'))

def main():
    """main function to generate business analysis and documentation"""
    print(colored('Starting business analysis pipeline...', 'blue'))
    
    try:
        #generate system architecture
        generate_system_architecture()
        
        #calculate theoretical business value
        business_value = calculate_business_value()
        
        #generate presentation materials
        generate_presentation_materials(business_value, None)
        
        print(colored('\nBusiness analysis pipeline completed successfully!', 'green'))
        print(colored('Documentation saved in documentation directory', 'blue'))
    except Exception as e:
        print(colored(f'\nBusiness analysis pipeline failed: {str(e)}', 'red'))
        raise

if __name__ == "__main__":
    main() 