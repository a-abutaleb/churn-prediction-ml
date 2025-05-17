# Telecom Customer Churn Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10.2-orange)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade MLOps project that predicts customer churn for a telecom company. The system uses MLflow for experiment tracking and model management, with a focus on scalability, monitoring, and maintainability.

## 🚀 Features

- **Multiple Models**: XGBoost, Random Forest, and Logistic Regression
- **MLflow Integration**: Experiment tracking, model registry, and model serving
- **REST API**: Flask-based API for model serving
- **Monitoring**: Real-time model and data drift monitoring
- **Testing**: Comprehensive test suite
- **Documentation**: Detailed project report and API documentation

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

## 🏃‍♂️ Quick Start

The setup script will:
1. Create necessary directories
2. Start the MLflow server
3. Train the models
4. Start the Flask API
5. Start the monitoring service

After running the setup script, you can access:
- MLflow UI: http://localhost:5050
- Flask API: http://localhost:5001
- Monitoring: http://localhost:8000

## 📚 Documentation

- [Project Report](docs/project_report.md): Detailed methodology and results
- [API Documentation](#api-endpoints): API usage and examples
- [Model Performance](#model-performance): Model metrics and analysis

## 🔌 API Endpoints

### 1. Health Check
```bash
curl http://localhost:5001/health
```
Response:
```json
{
  "model_loaded": true,
  "status": "healthy",
  "timestamp": "2025-05-17T03:33:53.262298"
}
```

### 2. Model Metadata
```bash
curl http://localhost:5001/metadata
```
Response:
```json
{
  "creation_timestamp": 1747441760460,
  "current_stage": "None",
  "description": "",
  "model_name": "churn_prediction_xgboost",
  "model_version": "1"
}
```

### 3. Predict Churn
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "tenure": 1,
    "monthly_charges": 29.85,
    "total_charges": 29.85
  }'
```
Response:
```json
{
  "churn_prediction": true,
  "churn_probability": 0.7734425663948059,
  "timestamp": "2025-05-17T03:32:53.924351"
}
```

## 📊 Model Performance

The best performing model is XGBoost with the following metrics:
- ROC AUC: 0.8441
- Accuracy: 0.7700
- Precision: 0.5512
- Recall: 0.7193
- F1 Score: 0.6241

## 📈 Monitoring

The project includes monitoring for:
- Data drift detection
- Model drift detection
- Performance metrics tracking
- Alert generation for anomalies

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

The test suite includes:
- API endpoint tests
- Input validation tests
- Model prediction tests
- Error handling tests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- MLflow team for the excellent MLOps platform
- XGBoost team for the powerful gradient boosting library
- Evidently team for the monitoring tools 