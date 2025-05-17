import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    DATA_PATH
)
from src.monitoring.monitor import ModelMonitor

def verify_monitoring():
    # Initialize the monitor
    monitor = ModelMonitor()
    
    # Load some test data
    data = pd.read_csv(DATA_PATH)
    X = data.drop('class', axis=1)
    
    # Make some predictions and monitor them
    print("\nMonitoring System Verification:")
    print("Making predictions and monitoring...")
    
    # Monitor predictions
    metrics = monitor.monitor_predictions(X)
    
    print("\nMonitoring Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Check if metrics are being saved
    metrics_path = "monitoring/metrics.json"
    if os.path.exists(metrics_path):
        print("\nMetrics file exists and is being updated")
    else:
        print("\nWarning: Metrics file not found")
    
    # Check data drift
    drift_score = monitor.calculate_data_drift(X)
    print(f"\nData Drift Score: {drift_score:.4f}")
    
    # Check if metrics are being logged to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="metrics.prediction_latency > 0"
    )
    
    if runs:
        print("\nMLflow logging is working")
        print(f"Found {len(runs)} runs with monitoring metrics")
    else:
        print("\nWarning: No monitoring metrics found in MLflow")

if __name__ == "__main__":
    verify_monitoring() 