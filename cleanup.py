"""
Script to clean up the churn prediction system.
"""

import os
import sys
import logging
import shutil
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def stop_servers():
    """Stop all running servers."""
    try:
        # Stop MLflow server
        logger.info("Stopping MLflow server...")
        subprocess.run(
            "pkill -f 'mlflow server'",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        
        # Stop Flask server
        logger.info("Stopping Flask server...")
        subprocess.run(
            "pkill -f 'python src/serve/api/app.py'",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        
        # Stop monitoring
        logger.info("Stopping monitoring system...")
        subprocess.run(
            "pkill -f 'python src/monitoring/run_monitoring.py'",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        
        # Stop Prometheus
        logger.info("Stopping Prometheus...")
        subprocess.run(
            "pkill -f 'prometheus'",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        
        # Stop Grafana
        logger.info("Stopping Grafana...")
        subprocess.run(
            "pkill -f 'grafana-server'",
            shell=True,
            stderr=subprocess.DEVNULL
        )
        
    except Exception as e:
        logger.error(f"Error stopping servers: {str(e)}")
        raise

def clean_directories():
    """Clean up generated directories and files."""
    try:
        # Directories to clean
        directories = [
            "data/processed",
            "models",
            "logs",
            "test_results",
            "monitoring/prometheus",
            "monitoring/grafana",
            "mlruns",
            "mlartifacts"
        ]
        
        # Files to clean
        files = [
            "mlflow.db",
            "confusion_matrix.png",
            "feature_importance.png",
            "classification_report.csv"
        ]
        
        # Remove directories
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logger.info(f"Removed directory: {directory}")
        
        # Remove files
        for file in files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed file: {file}")
        
    except Exception as e:
        logger.error(f"Error cleaning directories: {str(e)}")
        raise

def backup_data():
    """Backup important data before cleanup."""
    try:
        # Create backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backup_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Files to backup
        files_to_backup = [
            "data/processed/telecom_churn.csv",
            "models/best_model.pkl",
            "logs/training.log",
            "test_results/latest_results.txt"
        ]
        
        # Backup files
        for file in files_to_backup:
            if os.path.exists(file):
                # Create subdirectories in backup
                backup_path = os.path.join(backup_dir, file)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(file, backup_path)
                logger.info(f"Backed up: {file}")
        
        logger.info(f"Backup created in: {backup_dir}")
        
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        raise

def cleanup():
    """Clean up the entire system."""
    try:
        # Stop all servers
        stop_servers()
        
        # Backup important data
        backup_data()
        
        # Clean directories
        clean_directories()
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    cleanup() 