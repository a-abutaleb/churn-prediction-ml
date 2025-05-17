"""
Script to run the churn prediction system in production mode.
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

def create_prod_directories():
    """Create production directories."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "test_results",
        "monitoring/prometheus",
        "monitoring/grafana",
        "prod"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def start_mlflow_prod():
    """Start MLflow server in production mode."""
    try:
        logger.info("Starting MLflow server in production mode...")
        mlflow = subprocess.Popen(
            "mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///prod/mlflow.db",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return mlflow
    except Exception as e:
        logger.error(f"Error starting MLflow: {str(e)}")
        raise

def start_flask_prod():
    """Start Flask API server in production mode."""
    try:
        logger.info("Starting Flask API server in production mode...")
        flask = subprocess.Popen(
            "gunicorn --workers 4 --bind 0.0.0.0:5000 src.serve.api.app:app",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return flask
    except Exception as e:
        logger.error(f"Error starting Flask: {str(e)}")
        raise

def start_monitoring_prod():
    """Start monitoring system in production mode."""
    try:
        logger.info("Starting monitoring system in production mode...")
        monitoring = subprocess.Popen(
            "python src/monitoring/run_monitoring.py --prod",
            shell=True
        )
        time.sleep(5)  # Wait for system to start
        return monitoring
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise

def run_tests_prod():
    """Run test suite in production mode."""
    try:
        logger.info("Running tests in production mode...")
        result = subprocess.run(
            "python tests/run_tests.py --prod",
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

def run_prod():
    """Run the system in production mode."""
    try:
        # Create production directories
        create_prod_directories()
        
        # Start MLflow
        mlflow = start_mlflow_prod()
        
        # Run data processing
        logger.info("Running data processing in production mode...")
        subprocess.run(
            "python src/data/processing/process.py --prod",
            shell=True,
            check=True
        )
        
        # Run model training
        logger.info("Running model training in production mode...")
        subprocess.run(
            "python src/models/training/train.py --prod",
            shell=True,
            check=True
        )
        
        # Run model evaluation
        logger.info("Running model evaluation in production mode...")
        subprocess.run(
            "python src/models/evaluation/evaluate.py --prod",
            shell=True,
            check=True
        )
        
        # Start Flask API
        flask = start_flask_prod()
        
        # Start monitoring
        monitoring = start_monitoring_prod()
        
        # Run tests
        if not run_tests_prod():
            logger.warning("Tests failed, but continuing with system startup")
        
        # Keep the script running
        try:
            while True:
                time.sleep(60)
                logger.info("Production system is running...")
        except KeyboardInterrupt:
            logger.info("Stopping production system...")
            mlflow.terminate()
            flask.terminate()
            monitoring.terminate()
            logger.info("Production system stopped")
            
    except Exception as e:
        logger.error(f"Error in production system: {str(e)}")
        raise

if __name__ == "__main__":
    run_prod() 