#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed models reports/monitoring

# Start MLflow server
echo "Starting MLflow server..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5050 &

# Wait for MLflow server to start
echo "Waiting for MLflow server to start..."
sleep 5

# Train models
echo "Training models..."
python src/models/train.py

# Start Flask API
echo "Starting Flask API..."
python src/serve/app.py &

# Start monitoring
echo "Starting model monitoring..."
python src/monitoring/monitor.py &

echo "Setup complete! The following services are running:"
echo "- MLflow server: http://localhost:5050"
echo "- Flask API: http://localhost:5001"
echo "- Model monitoring: http://localhost:8000"

# Keep script running
wait 