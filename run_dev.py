"""
Script to run the churn prediction system in development mode.
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dev_directories():
    """Create development directories."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "test_results",
        "monitoring/prometheus",
        "monitoring/grafana",
        "dev"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def start_mlflow_dev():
    """Start MLflow server in development mode."""
    try:
        logger.info("Starting MLflow server in development mode...")
        mlflow = subprocess.Popen(
            "mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///dev/mlflow.db",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return mlflow
    except Exception as e:
        logger.error(f"Error starting MLflow: {str(e)}")
        raise

def start_flask_dev():
    """Start Flask API server in development mode."""
    try:
        logger.info("Starting Flask API server in development mode...")
        flask = subprocess.Popen(
            "FLASK_ENV=development FLASK_DEBUG=1 python src/serve/api/app.py",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return flask
    except Exception as e:
        logger.error(f"Error starting Flask: {str(e)}")
        raise

def start_monitoring_dev():
    """Start monitoring system in development mode."""
    try:
        logger.info("Starting monitoring system in development mode...")
        monitoring = subprocess.Popen(
            "python src/monitoring/run_monitoring.py --dev",
            shell=True
        )
        time.sleep(5)  # Wait for system to start
        return monitoring
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise

def run_tests_dev():
    """Run test suite in development mode."""
    try:
        logger.info("Running tests in development mode...")
        result = subprocess.run(
            "python tests/run_tests.py --dev",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {str(e)}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_dev():
    """Run the system in development mode."""
    try:
        # Create development directories
        create_dev_directories()
        
        # Start MLflow
        mlflow = start_mlflow_dev()
        
        # Run data processing
        logger.info("Running data processing in development mode...")
        subprocess.run(
            "python src/data/processing/process.py --dev",
            shell=True,
            check=True
        )
        
        # Run model training
        logger.info("Running model training in development mode...")
        subprocess.run(
            "python src/models/training/train.py --dev",
            shell=True,
            check=True
        )
        
        # Run model evaluation
        logger.info("Running model evaluation in development mode...")
        subprocess.run(
            "python src/models/evaluation/evaluate.py --dev",
            shell=True,
            check=True
        )
        
        # Start Flask API
        flask = start_flask_dev()
        
        # Start monitoring
        monitoring = start_monitoring_dev()
        
        # Run tests
        if not run_tests_dev():
            logger.warning("Tests failed, but continuing with system startup")
        
        # Keep the script running
        try:
            while True:
                time.sleep(60)
                logger.info("Development system is running...")
        except KeyboardInterrupt:
            logger.info("Stopping development system...")
            mlflow.terminate()
            flask.terminate()
            monitoring.terminate()
            logger.info("Development system stopped")
            
    except Exception as e:
        logger.error(f"Error in development system: {str(e)}")
        raise

if __name__ == "__main__":
    run_dev() 