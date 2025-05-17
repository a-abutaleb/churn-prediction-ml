"""
Monitoring script for detecting data and model drift in the churn prediction model.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score
from scipy import stats
import prometheus_client as prom
from prometheus_client import start_http_server

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    DRIFT_THRESHOLD,
    MONITORING_INTERVAL,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
prediction_latency = prom.Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

prediction_requests = prom.Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['status']
)

feature_drift = prom.Gauge(
    'feature_drift_score',
    'Drift score for each feature',
    ['feature']
)

model_drift = prom.Gauge(
    'model_drift_score',
    'Overall model drift score'
)

def load_model():
    """Load the latest production model from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{model_version.version}")
        logger.info(f"Loaded model version {model_version.version}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_reference_data():
    """Load reference data used for training."""
    try:
        data = pd.read_csv("data/processed/telecom_churn.csv")
        logger.info(f"Loaded reference data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading reference data: {str(e)}")
        raise

def calculate_feature_drift(reference_data, current_data, feature):
    """Calculate drift score for a single feature."""
    if feature in CATEGORICAL_FEATURES:
        # For categorical features, use chi-square test
        ref_counts = reference_data[feature].value_counts(normalize=True)
        curr_counts = current_data[feature].value_counts(normalize=True)
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_counts = ref_counts.reindex(all_categories, fill_value=0)
        curr_counts = curr_counts.reindex(all_categories, fill_value=0)
        
        # Calculate chi-square statistic
        chi2, p_value = stats.chisquare(curr_counts, ref_counts)
        return 1 - p_value
    else:
        # For numerical features, use Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(
            reference_data[feature],
            current_data[feature]
        )
        return 1 - p_value

def detect_drift(reference_data, current_data, model):
    """Detect data and model drift."""
    drift_scores = {}
    
    # Calculate feature drift
    for feature in CATEGORICAL_FEATURES + NUMERICAL_FEATURES:
        drift_score = calculate_feature_drift(reference_data, current_data, feature)
        drift_scores[feature] = drift_score
        feature_drift.labels(feature=feature).set(drift_score)
    
    # Calculate model drift (using ROC AUC on current data)
    y_true = current_data['churn']
    y_pred = model.predict_proba(current_data.drop('churn', axis=1))[:, 1]
    model_score = roc_auc_score(y_true, y_pred)
    model_drift.set(model_score)
    
    return drift_scores, model_score

def monitor_drift():
    """Main monitoring function."""
    try:
        # Load model and reference data
        model = load_model()
        reference_data = load_reference_data()
        
        # Start Prometheus metrics server
        start_http_server(8000)
        logger.info("Started Prometheus metrics server on port 8000")
        
        while True:
            try:
                # Load current data (in a real scenario, this would be streaming data)
                current_data = pd.read_csv("data/processed/current_data.csv")
                
                # Detect drift
                drift_scores, model_score = detect_drift(reference_data, current_data, model)
                
                # Log drift scores
                logger.info("Feature Drift Scores:")
                for feature, score in drift_scores.items():
                    logger.info(f"{feature}: {score:.4f}")
                logger.info(f"Model Score: {model_score:.4f}")
                
                # Check for significant drift
                significant_drift = any(score > DRIFT_THRESHOLD for score in drift_scores.values())
                if significant_drift:
                    logger.warning("Significant drift detected!")
                    
                    # In a real scenario, you might want to:
                    # 1. Send alerts
                    # 2. Trigger model retraining
                    # 3. Update reference data
                
                # Wait for next monitoring interval
                time.sleep(MONITORING_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(MONITORING_INTERVAL)
                
    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")
        raise

if __name__ == "__main__":
    monitor_drift() 