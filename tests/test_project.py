import pytest
import os
import sys
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import mlflow
import json
from pathlib import Path
from src.models.train import train_models, evaluate_model, load_data
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.serve import app
from src.monitoring.monitor import ModelMonitor

# Initialize test client
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_mlflow_for_tests():
    """Setup MLflow for testing"""
    setup_mlflow()

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = pd.DataFrame({
        'fixed_acidity': [7.4, 7.8, 7.3],
        'volatile_acidity': [0.7, 0.88, 0.65],
        'citric_acid': [0, 0, 0.1],
        'residual_sugar': [1.9, 2.6, 1.2],
        'chlorides': [0.076, 0.098, 0.065],
        'free_sulfur_dioxide': [11, 25, 15],
        'total_sulfur_dioxide': [34, 67, 21],
        'density': [0.9978, 0.9968, 0.9959],
        'pH': [3.51, 3.2, 3.3],
        'sulphates': [0.56, 0.68, 0.47],
        'alcohol': [9.4, 9.8, 10]
    })
    y = pd.Series([5, 5, 6])
    return X, y

def test_data_loading():
    """Test data loading functionality"""
    X_train, X_test, y_train, y_test = load_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) > 0
    assert len(y_train) > 0

def test_model_training():
    """Test model training functionality"""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = preprocess_data(X_train)
    models = train_models(X_train, y_train, preprocessor)
    
    # Check if all models are trained
    model_names = [name for name, _, _ in models]
    assert 'xgboost' in model_names
    assert 'random_forest' in model_names
    assert 'logistic_regression' in model_names
    
    # Check if models are fitted pipelines
    for _, model, _ in models:
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

def test_model_evaluation():
    """Test model evaluation metrics"""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = preprocess_data(X_train)
    models = train_models(X_train, y_train, preprocessor)
    
    for name, model, _ in models:
        metrics, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
        # Check if all metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

def test_api_health():
    """Test API health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_api_prediction():
    """Test API prediction endpoint"""
    test_data = {
        "fixed_acidity": [7.4],
        "volatile_acidity": [0.7],
        "citric_acid": [0],
        "residual_sugar": [1.9],
        "chlorides": [0.076],
        "free_sulfur_dioxide": [11],
        "total_sulfur_dioxide": [34],
        "density": [0.9978],
        "pH": [3.51],
        "sulphates": [0.56],
        "alcohol": [9.4]
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

class TestModelMonitor:
    @pytest.fixture
    def monitor(self):
        return ModelMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.model is not None
        assert isinstance(monitor.reference_data, pd.DataFrame)
        assert isinstance(monitor.metrics_history, dict)
    
    def test_data_drift_calculation(self, monitor, sample_data):
        """Test data drift calculation"""
        X, _ = sample_data
        drift_score = monitor.calculate_data_drift(X)
        assert isinstance(drift_score, float)
        assert 0 <= drift_score <= 1
    
    def test_monitoring_predictions(self, monitor, sample_data):
        """Test monitoring predictions"""
        X, _ = sample_data
        metrics = monitor.monitor_predictions(X)
        
        assert isinstance(metrics, dict)
        assert 'prediction_latency' in metrics
        assert 'data_drift_score' in metrics
        assert 'prediction_mean' in metrics
        assert 'prediction_std' in metrics
    
    def test_metrics_storage(self, monitor, sample_data, tmp_path):
        """Test metrics storage functionality"""
        X, _ = sample_data
        test_metrics_path = tmp_path / "test_metrics.json"
        
        # Monitor predictions and save metrics
        monitor.monitor_predictions(X)
        monitor.save_metrics_history(str(test_metrics_path))
        
        # Verify metrics file exists and is valid JSON
        assert test_metrics_path.exists()
        with open(test_metrics_path) as f:
            metrics_data = json.load(f)
        
        assert isinstance(metrics_data, dict)
        assert len(metrics_data['timestamp']) > 0

if __name__ == '__main__':
    pytest.main([__file__]) 