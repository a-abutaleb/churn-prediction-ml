"""
Main script to run the entire ML pipeline.
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

def run_command(command, description):
    """Run a shell command and log its output."""
    logger.info(f"Starting: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Completed: {description}")
        logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {description}: {str(e)}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_pipeline():
    """Run the entire ML pipeline."""
    try:
        # Create necessary directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Start MLflow server
        logger.info("Starting MLflow server...")
        mlflow_server = subprocess.Popen(
            "mlflow server --host 0.0.0.0 --port 5050",
            shell=True
        )
        time.sleep(5)  # Wait for server to start

        # Run data processing
        if not run_command(
            "python src/data/processing/process.py",
            "Data processing"
        ):
            raise Exception("Data processing failed")

        # Run model training
        if not run_command(
            "python src/models/training/train.py",
            "Model training"
        ):
            raise Exception("Model training failed")

        # Run model evaluation
        if not run_command(
            "python src/models/evaluation/evaluate.py",
            "Model evaluation"
        ):
            raise Exception("Model evaluation failed")

        # Start model serving
        logger.info("Starting model serving...")
        flask_server = subprocess.Popen(
            "python src/serve/api/app.py",
            shell=True
        )
        time.sleep(5)  # Wait for server to start

        # Start monitoring
        logger.info("Starting model monitoring...")
        monitor = subprocess.Popen(
            "python src/monitoring/drift/monitor.py",
            shell=True
        )

        # Keep the script running
        try:
            while True:
                time.sleep(60)
                logger.info("Pipeline is running...")
        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
            mlflow_server.terminate()
            flask_server.terminate()
            monitor.terminate()
            logger.info("Pipeline stopped")

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline() 