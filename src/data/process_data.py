import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from config.mlflow_config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def load_data(file_path):
    """Load data from file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data: handle missing values and encode categoricals."""
    df = df.copy()
    # Drop rows with missing target
    if 'churn' in df.columns:
        df = df.dropna(subset=['churn'])
    # Fill missing numeric values with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    # One-hot encode categoricals (drop first to avoid collinearity)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def process_data(file_path, target_column):
    """Main function to process data and log to MLflow."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="data_processing"):
        # Load data
        df = load_data(file_path)
        
        # Log data statistics
        mlflow.log_metric("num_samples", len(df))
        mlflow.log_metric("num_features", len(df.columns) - 1)  # excluding target
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df_processed, target_column)
        
        # Log split statistics
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)
        
        # Log artifacts
        mlflow.log_artifacts("data/processed", "processed_data")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    file_path = "data/raw/your_dataset.csv"  # Replace with your dataset path
    target_column = "target"  # Replace with your target column name
    process_data(file_path, target_column) 