import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    DATA_PATH
)

def verify_predictions():
    # Load the model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    
    # Load the test data
    data = pd.read_csv(DATA_PATH)
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Print some example predictions
    print("\nExample Predictions vs Actual Values:")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  Features: {X.iloc[i].to_dict()}")
        print(f"  Actual Quality: {y.iloc[i]}")
        print(f"  Predicted Quality: {predictions[i]:.2f}")
        print()

if __name__ == "__main__":
    verify_predictions() 