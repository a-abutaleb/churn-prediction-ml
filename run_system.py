"""
Script to run the entire churn prediction system.
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

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "test_results",
        "monitoring/prometheus",
        "monitoring/grafana"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def start_mlflow():
    """Start MLflow server."""
    try:
        logger.info("Starting MLflow server...")
        mlflow = subprocess.Popen(
            "mlflow server --host 0.0.0.0 --port 5050",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return mlflow
    except Exception as e:
        logger.error(f"Error starting MLflow: {str(e)}")
        raise

def start_flask():
    """Start Flask API server."""
    try:
        logger.info("Starting Flask API server...")
        flask = subprocess.Popen(
            "python src/serve/api/app.py",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return flask
    except Exception as e:
        logger.error(f"Error starting Flask: {str(e)}")
        raise

def start_monitoring():
    """Start monitoring system."""
    try:
        logger.info("Starting monitoring system...")
        monitoring = subprocess.Popen(
            "python src/monitoring/run_monitoring.py",
            shell=True
        )
        time.sleep(5)  # Wait for system to start
        return monitoring
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise

def run_tests():
    """Run test suite."""
    try:
        logger.info("Running tests...")
        result = subprocess.run(
            "python tests/run_tests.py",
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

def run_system():
    """Run the entire system."""
    try:
        # Create directories
        create_directories()
        
        # Start MLflow
        mlflow = start_mlflow()
        
        # Run data processing
        logger.info("Running data processing...")
        subprocess.run(
            "python src/data/processing/process.py",
            shell=True,
            check=True
        )
        
        # Run model training
        logger.info("Running model training...")
        subprocess.run(
            "python src/models/training/train.py",
            shell=True,
            check=True
        )
        
        # Run model evaluation
        logger.info("Running model evaluation...")
        subprocess.run(
            "python src/models/evaluation/evaluate.py",
            shell=True,
            check=True
        )
        
        # Start Flask API
        flask = start_flask()
        
        # Start monitoring
        monitoring = start_monitoring()
        
        # Run tests
        if not run_tests():
            logger.warning("Tests failed, but continuing with system startup")
        
        # Keep the script running
        try:
            while True:
                time.sleep(60)
                logger.info("System is running...")
        except KeyboardInterrupt:
            logger.info("Stopping system...")
            mlflow.terminate()
            flask.terminate()
            monitoring.terminate()
            logger.info("System stopped")
            
    except Exception as e:
        logger.error(f"Error in system: {str(e)}")
        raise

if __name__ == "__main__":
    run_system() 