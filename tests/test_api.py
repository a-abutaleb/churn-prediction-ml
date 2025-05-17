"""
Test script for the churn prediction API endpoints.
"""

import os
import sys
import json
import unittest
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from config.mlflow_config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)

class TestChurnPredictionAPI(unittest.TestCase):
    """Test cases for the churn prediction API."""

    def setUp(self):
        """Set up test environment."""
        self.base_url = "http://localhost:5000"
        self.test_data = self._generate_test_data()

    def _generate_test_data(self):
        """Generate test data for API requests."""
        # Load a sample from the processed data
        try:
            data = pd.read_csv("data/processed/telecom_churn.csv")
            return data.iloc[0].to_dict()
        except Exception:
            # Generate synthetic data if file not found
            return {
                # Categorical features
                'gender': 'Male',
                'senior_citizen': 0,
                'partner': 'Yes',
                'dependents': 'No',
                'phone_service': 'Yes',
                'multiple_lines': 'No',
                'internet_service': 'DSL',
                'online_security': 'No',
                'online_backup': 'Yes',
                'device_protection': 'No',
                'tech_support': 'No',
                'streaming_tv': 'No',
                'streaming_movies': 'No',
                'contract': 'Month-to-month',
                'paperless_billing': 'Yes',
                'payment_method': 'Electronic check',
                
                # Numerical features
                'tenure': 1,
                'monthly_charges': 29.85,
                'total_charges': 29.85
            }

    def test_health_check(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_model_info(self):
        """Test model info endpoint."""
        response = requests.get(f"{self.base_url}/model-info")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("model_name", data)
        self.assertIn("version", data)
        self.assertIn("stage", data)
        self.assertIn("features", data)
        
        # Check features
        self.assertIn("categorical", data["features"])
        self.assertIn("numerical", data["features"])
        self.assertEqual(data["features"]["categorical"], CATEGORICAL_FEATURES)
        self.assertEqual(data["features"]["numerical"], NUMERICAL_FEATURES)

    def test_predict_valid_data(self):
        """Test prediction endpoint with valid data."""
        response = requests.post(
            f"{self.base_url}/predict",
            json=self.test_data
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("model_version", data)
        
        # Check prediction values
        self.assertIn(data["prediction"], [0, 1])
        self.assertTrue(0 <= data["probability"] <= 1)

    def test_predict_missing_features(self):
        """Test prediction endpoint with missing features."""
        # Remove a required feature
        invalid_data = self.test_data.copy()
        del invalid_data["tenure"]
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=invalid_data
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertIn("Missing required features", response.json()["error"])

    def test_predict_invalid_data(self):
        """Test prediction endpoint with invalid data."""
        # Send empty data
        response = requests.post(
            f"{self.base_url}/predict",
            json={}
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertIn("No input data provided", response.json()["error"])

    def test_predict_invalid_values(self):
        """Test prediction endpoint with invalid feature values."""
        # Modify a numerical feature with invalid value
        invalid_data = self.test_data.copy()
        invalid_data["tenure"] = "invalid"
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=invalid_data
        )
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())

def run_tests():
    """Run the test suite."""
    unittest.main()

if __name__ == "__main__":
    run_tests() 