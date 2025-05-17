"""
Flask API for serving the churn prediction model.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model
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

# Load model at startup
model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Validate required features
        required_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        # Prepare response
        response = {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0]),
            "model_version": model.version
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information endpoint."""
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        
        info = {
            "model_name": MODEL_NAME,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            "features": {
                "categorical": CATEGORICAL_FEATURES,
                "numerical": NUMERICAL_FEATURES
            }
        }
        
        return jsonify(info), 200

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 