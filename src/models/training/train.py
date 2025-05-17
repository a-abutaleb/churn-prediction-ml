"""
Training script for the churn prediction model using MLflow for experiment tracking.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_NAME,
    RANDOM_SEED,
    TEST_SIZE,
    CV_FOLDS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    METRICS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load and preprocess the data."""
    try:
        data = pd.read_csv("data/processed/telecom_churn.csv")
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_preprocessing_pipeline():
    """Create preprocessing pipeline for numerical and categorical features."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    return preprocessor

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    with mlflow.start_run(nested=True):
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5)
        }

        # Log parameters
        mlflow.log_params(params)

        # Create model with parameters
        model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED)

        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline()),
            ('classifier', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {}
        for metric_name in METRICS:
            if metric_name == 'roc_auc':
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                score = getattr(metrics, metric_name)(y_test, y_pred)
            metrics[metric_name] = score
            mlflow.log_metric(metric_name, score)

        return metrics['roc_auc']

def train_model():
    """Train the model using MLflow for experiment tracking."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model_training"):
        # Load and prepare data
        data = load_data()
        X = data.drop(TARGET_COLUMN, axis=1)
        y = data[TARGET_COLUMN]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # Log data info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", list(X.columns))

        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)

        # Get best parameters
        best_params = study.best_params
        mlflow.log_params(best_params)

        # Train final model with best parameters
        final_model = xgb.XGBClassifier(**best_params, random_state=RANDOM_SEED)
        pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline()),
            ('classifier', final_model)
        ])

        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Log metrics
        for metric_name in METRICS:
            if metric_name == 'roc_auc':
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                score = getattr(metrics, metric_name)(y_test, y_pred)
            mlflow.log_metric(metric_name, score)

        # Log model
        signature = infer_signature(X_test, y_pred)
        mlflow.xgboost.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature
        )

        logger.info("Model training completed successfully")
        return pipeline

if __name__ == "__main__":
    try:
        model = train_model()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise 