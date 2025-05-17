#!/bin/bash

# Test health check endpoint
echo "Testing health check endpoint..."
curl -X GET http://localhost:8000/health

# Test prediction endpoint with sample data
echo -e "\n\nTesting prediction endpoint..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Test prediction endpoint with invalid data
echo -e "\n\nTesting prediction endpoint with invalid data..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": "invalid",
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
  }' 