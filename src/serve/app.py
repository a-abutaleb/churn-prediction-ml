from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
from pathlib import Path
from src.monitoring.monitor import ModelMonitor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "telecom_churn_prediction")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create Flask app
app = Flask(__name__)

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Load reference data for monitoring
try:
    reference_data = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Initialize model monitor
    model_monitor = ModelMonitor("churn_prediction_xgboost", reference_data)
    logger.info("Model monitor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model monitor: {str(e)}")
    model_monitor = None

# Define feature columns
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

def load_latest_model():
    """Load the latest model from MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions("name='churn_prediction_xgboost'")
        if not model_versions:
            logger.warning("No model versions found, using dummy model")
            return create_dummy_model()
        latest_version = max(model_versions, key=lambda x: x.version)
        model = mlflow.sklearn.load_model(f"models:/churn_prediction_xgboost/{latest_version.version}")
        logger.info(f"Loaded model version {latest_version.version}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("Using dummy model due to MLflow connection error")
        return create_dummy_model()

def create_dummy_model():
    """Create a dummy model for testing"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Train on a small dummy dataset
    X = np.random.rand(10, len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    return model

model = load_latest_model()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })

@app.route("/metadata", methods=["GET"])
def get_model_metadata():
    """Get model metadata"""
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions("name='churn_prediction_xgboost'")
        
        if not model_versions:
            return jsonify({
                "model_name": "dummy_churn_prediction_model",
                "model_version": "1",
                "creation_timestamp": datetime.now().isoformat(),
                "current_stage": "Testing",
                "description": "Dummy model for testing purposes"
            })
        
        latest_version = max(model_versions, key=lambda x: x.version)
        
        return jsonify({
            "model_name": latest_version.name,
            "model_version": latest_version.version,
            "creation_timestamp": latest_version.creation_timestamp,
            "current_stage": latest_version.current_stage,
            "description": latest_version.description
        })
    except Exception as e:
        logger.error(f"Error getting model metadata: {str(e)}")
        return jsonify({
            "model_name": "dummy_churn_prediction_model",
            "model_version": "1",
            "creation_timestamp": datetime.now().isoformat(),
            "current_stage": "Testing",
            "description": "Dummy model for testing purposes"
        })

@app.route("/predict", methods=["POST"])
def predict():
    """Make churn predictions"""
    try:
        data = request.get_json()
        
        # Map input field names to model feature names
        field_mapping = {
            'gender': 'gender',
            'partner': 'Partner',
            'dependents': 'Dependents',
            'phone_service': 'PhoneService',
            'multiple_lines': 'MultipleLines',
            'internet_service': 'InternetService',
            'online_security': 'OnlineSecurity',
            'online_backup': 'OnlineBackup',
            'device_protection': 'DeviceProtection',
            'tech_support': 'TechSupport',
            'streaming_tv': 'StreamingTV',
            'streaming_movies': 'StreamingMovies',
            'contract': 'Contract',
            'paperless_billing': 'PaperlessBilling',
            'payment_method': 'PaymentMethod',
            'tenure': 'tenure',
            'monthly_charges': 'MonthlyCharges',
            'total_charges': 'TotalCharges'
        }
        
        # Validate required fields
        for field in field_mapping.keys():
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Validate numeric fields
        if not isinstance(data["tenure"], (int, float)) or data["tenure"] < 0:
            return jsonify({"error": "tenure must be a non-negative number"}), 400
        if not isinstance(data["monthly_charges"], (int, float)) or data["monthly_charges"] < 0:
            return jsonify({"error": "monthly_charges must be a non-negative number"}), 400
        if not isinstance(data["total_charges"], (int, float)) or data["total_charges"] < 0:
            return jsonify({"error": "total_charges must be a non-negative number"}), 400
        
        # Create DataFrame with correct column names
        features = pd.DataFrame({
            model_col: [data[api_col]] for api_col, model_col in field_mapping.items()
        })
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Monitor prediction if monitor is initialized
        if model_monitor is not None:
            monitoring_results = model_monitor.monitor_prediction(features, prediction)
            logger.info(f"Monitoring results: {monitoring_results}")
        
        return jsonify({
            "churn_prediction": bool(prediction),
            "churn_probability": float(probability),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Make batch churn predictions"""
    try:
        data = request.get_json()
        if 'data' not in data:
            return jsonify({"error": "Missing 'data' field"}), 400
        
        predictions = []
        for input_data in data['data']:
            # Create DataFrame for single prediction
            field_mapping = {
                'gender': 'gender',
                'partner': 'Partner',
                'dependents': 'Dependents',
                'phone_service': 'PhoneService',
                'multiple_lines': 'MultipleLines',
                'internet_service': 'InternetService',
                'online_security': 'OnlineSecurity',
                'online_backup': 'OnlineBackup',
                'device_protection': 'DeviceProtection',
                'tech_support': 'TechSupport',
                'streaming_tv': 'StreamingTV',
                'streaming_movies': 'StreamingMovies',
                'contract': 'Contract',
                'paperless_billing': 'PaperlessBilling',
                'payment_method': 'PaymentMethod',
                'tenure': 'tenure',
                'monthly_charges': 'MonthlyCharges',
                'total_charges': 'TotalCharges'
            }
            
            # Validate required fields
            for field in field_mapping.keys():
                if field not in input_data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Create DataFrame
            features = pd.DataFrame({
                model_col: [input_data[api_col]] 
                for api_col, model_col in field_mapping.items()
            })
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            predictions.append({
                "churn_prediction": bool(prediction),
                "churn_probability": float(probability),
                "timestamp": datetime.now().isoformat()
            })
        
        return jsonify({"predictions": predictions})
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
