import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from src.models.serve import app, CustomerData

# Create test client
client = TestClient(app)

# Sample test data
test_customer = {
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "gender": "Female",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "No",
    "device_protection": "Yes",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check"
}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data."""
    response = client.post("/predict", json=test_customer)
    assert response.status_code == 200
    
    result = response.json()
    assert "churn_probability" in result
    assert "prediction" in result
    assert "features_importance" in result
    
    # Check probability is between 0 and 1
    assert 0 <= result["churn_probability"] <= 1
    
    # Check prediction is either "Yes" or "No"
    assert result["prediction"] in ["Yes", "No"]
    
    # Check feature importance is a dictionary
    assert isinstance(result["features_importance"], dict)

def test_invalid_data():
    """Test the prediction endpoint with invalid data."""
    invalid_data = test_customer.copy()
    invalid_data["monthly_charges"] = "invalid"  # Should be a number
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 