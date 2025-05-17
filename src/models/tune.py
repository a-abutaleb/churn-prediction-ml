import os
import mlflow
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml_lifecycle_project")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_data():
    """Load training and test data."""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def objective(params):
    """Objective function for hyperparameter optimization."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_params(params)

        # Load data
        X_train, X_test, y_train, y_test = load_data()

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        return {'loss': -accuracy, 'status': STATUS_OK}

def tune_hyperparameters(max_evals=5):
    """Tune hyperparameters using Hyperopt."""
    # Define search space
    space = {
        'n_estimators': hp.choice('n_estimators', [5, 10, 20]),
        'max_depth': hp.choice('max_depth', [2, 3, 5]),
        'min_samples_split': hp.choice('min_samples_split', [2, 3, 4]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 3]),
        'random_state': 42
    }

    # Run optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    # Convert best parameters to actual values
    best_params = {
        'n_estimators': [5, 10, 20][best['n_estimators']],
        'max_depth': [2, 3, 5][best['max_depth']],
        'min_samples_split': [2, 3, 4][best['min_samples_split']],
        'min_samples_leaf': [1, 2, 3][best['min_samples_leaf']],
        'random_state': 42
    }

    return best_params

if __name__ == "__main__":
    # Run hyperparameter tuning
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 