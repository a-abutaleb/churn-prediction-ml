"""
Evaluation script for the churn prediction model using MLflow for tracking metrics.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_NAME,
    RANDOM_SEED,
    TEST_SIZE,
    METRICS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the test data."""
    try:
        data = pd.read_csv("data/processed/telecom_churn.csv")
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

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

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics to MLflow."""
    with mlflow.start_run(run_name="model_evaluation"):
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        # Generate and log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv')
        mlflow.log_artifact('classification_report.csv')

        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()

        return metrics

def main():
    """Main evaluation function."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load data
        data = load_data()
        X = data.drop('churn', axis=1)
        y = data['churn']

        # Load model
        model = load_model()

        # Evaluate model
        metrics = evaluate_model(model, X, y)
        
        # Print metrics
        logger.info("Model Evaluation Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 