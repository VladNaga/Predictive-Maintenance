import os
from termcolor import colored
import subprocess

def run_script(script_name):
    """Run a Python script and handle its output"""
    print(colored(f'\nRunning {script_name}...', 'blue'))
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print(colored(f'{script_name} completed successfully', 'green'))
            return True
        else:
            print(colored(f'Error in {script_name}:', 'red'))
            print(result.stderr)
            return False
    except Exception as e:
        print(colored(f'Failed to run {script_name}: {str(e)}', 'red'))
        return False

def main():
    """Main function to run the complete pipeline"""
    print(colored('Starting predictive maintenance pipeline...\n', 'blue'))
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('analysis_output', exist_ok=True)
    os.makedirs('documentation', exist_ok=True)
    
    # Run data preparation
    if not run_script('src/data_preparation.py'):
        return
    
    # Run model training
    if not run_script('src/model_training.py'):
        return
    
    # Run model analysis
    if not run_script('src/model_analysis.py'):
        return
    
    # Run business analysis
    if not run_script('src/business_analysis.py'):
        return
    
    print(colored('\nPipeline completed successfully!', 'green'))
    print(colored('Results saved in:', 'blue'))
    print(colored('- data/processed: Processed data files', 'blue'))
    print(colored('- models: Trained model files', 'blue'))
    print(colored('- analysis_output: Analysis results and visualizations', 'blue'))
    print(colored('- documentation: System architecture and business analysis', 'blue'))

if __name__ == "__main__":
    main() 