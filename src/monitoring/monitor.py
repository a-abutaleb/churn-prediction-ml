import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import sys
from scipy.stats import ks_2samp
import logging
import json
from typing import Dict, Any, Optional
import requests
from pathlib import Path
from urllib.parse import urlparse
from .drift_detector import DriftDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from prometheus_client import start_http_server, Gauge, Counter
from mlflow.tracking import MlflowClient

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    DATA_PATH
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_name', 'prediction']
)
PREDICTION_LATENCY = Gauge(
    'model_prediction_latency_seconds',
    'Time taken to make predictions',
    ['model_name']
)
DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Data drift score between reference and current data',
    ['feature']
)

class ModelMonitor:
    def __init__(self, reference_data_path, mlflow_tracking_uri):
        """Initialize the model monitor"""
        self.reference_data = pd.read_csv(reference_data_path)
        self.mlflow_client = MlflowClient(mlflow_tracking_uri)
        self.current_model = None
        self.current_model_version = None
        
    def load_current_model(self, model_name, version=None):
        """Load the current production model from MLflow"""
        try:
            if version:
                model_details = self.mlflow_client.get_model_version(model_name, version)
            else:
                model_details = self.mlflow_client.get_latest_versions(model_name, stages=["Production"])[0]
            
            self.current_model = mlflow.pyfunc.load_model(model_details.source)
            self.current_model_version = model_details.version
            logger.info(f"Loaded model {model_name} version {self.current_model_version}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def check_data_drift(self, current_data):
        """Check for data drift between reference and current data"""
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=self.reference_data, current_data=current_data)
            
            # Update Prometheus metrics
            for feature, drift_score in report.get_metrics()['data_drift']['drift_by_feature'].items():
                DATA_DRIFT_SCORE.labels(feature=feature).set(drift_score)
            
            return report.get_metrics()
        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}")
            return None

    def evaluate_model_performance(self, current_data, predictions):
        """Evaluate model performance on current data"""
        try:
            report = Report(metrics=[ClassificationPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping={
                    'prediction': 'prediction',
                    'target': 'churn'
                }
            )
            return report.get_metrics()
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            return None

    def log_prediction(self, prediction, prediction_time):
        """Log prediction metrics"""
        PREDICTION_COUNTER.labels(
            model_name=self.current_model_version,
            prediction=str(prediction)
        ).inc()
        PREDICTION_LATENCY.labels(
            model_name=self.current_model_version
        ).set(prediction_time)

    def save_monitoring_report(self, report, output_path):
        """Save monitoring report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(output_path, f"monitoring_report_{timestamp}.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Saved monitoring report to {report_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving monitoring report: {str(e)}")
            return False

def main():
    # Initialize monitor
    monitor = ModelMonitor(
        reference_data_path="data/processed/reference_data.csv",
        mlflow_tracking_uri="http://localhost:5050"
    )
    
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Load current model
    if not monitor.load_current_model("churn_prediction_xgboost"):
        logger.error("Failed to load model")
        return
    
    # Example monitoring loop
    while True:
        try:
            # Load current data (implement your data loading logic)
            current_data = pd.read_csv("data/processed/current_data.csv")
            
            # Check for data drift
            drift_report = monitor.check_data_drift(current_data)
            if drift_report:
                logger.info("Data drift detected")
                monitor.save_monitoring_report(drift_report, "reports/monitoring")
            
            # Evaluate model performance
            predictions = monitor.current_model.predict(current_data)
            performance_report = monitor.evaluate_model_performance(current_data, predictions)
            if performance_report:
                logger.info("Model performance evaluated")
                monitor.save_monitoring_report(performance_report, "reports/monitoring")
            
            # Sleep for monitoring interval
            time.sleep(3600)  # Check every hour
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    main() 