"""
MLflow configuration settings for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# MLflow settings
MLFLOW_TRACKING_URI = "http://localhost:5050"
EXPERIMENT_NAME = "churn_prediction"
MODEL_NAME = "churn_prediction_xgboost"

# Model registry settings
STAGING_MODEL_NAME = f"{MODEL_NAME}_staging"
PRODUCTION_MODEL_NAME = f"{MODEL_NAME}_production"

# Training settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature settings
CATEGORICAL_FEATURES = [
    'gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
    'internet_service', 'online_security', 'online_backup', 'device_protection',
    'tech_support', 'streaming_tv', 'streaming_movies', 'contract',
    'paperless_billing', 'payment_method'
]

NUMERICAL_FEATURES = ['tenure', 'monthly_charges', 'total_charges']

TARGET_COLUMN = 'churn'

# Hyperparameter tuning settings
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour

# Model evaluation metrics
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Monitoring settings
DRIFT_THRESHOLD = 0.05
MONITORING_INTERVAL = 3600  # 1 hour

# Model Configuration
MODEL_VERSION = "1.0.0"

# Data Configuration
DATA_PATH = "data/winequality-red.csv"

# Model Configuration
MODEL_DIR = "models"
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "artifacts")

# Model Registry Configuration
STAGING_STAGE = "Staging"
PRODUCTION_STAGE = "Production"
ARCHIVED_STAGE = "Archived"

# Model Registry Configuration
MODEL_REGISTRY_NAME = "ml_lifecycle_models"

# Data Configuration
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed") 