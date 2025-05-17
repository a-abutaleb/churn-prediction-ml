import pytest
import numpy as np
from src.models.train import load_data, preprocess_data, train_models, evaluate_model
from sklearn.model_selection import train_test_split
import joblib

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

def test_model_persistence():
    """Test model saving and loading (best model only)"""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = preprocess_data(X_train)
    models = train_models(X_train, y_train, preprocessor)
    # Find best model by ROC AUC
    best_model = None
    best_score = -np.inf
    for name, model, _ in models:
        metrics, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model = model
    # Save and load
    joblib.dump(best_model, 'models/test_churn_model.joblib')
    loaded_model = joblib.load('models/test_churn_model.joblib')
    # Check if predictions match
    pred1 = best_model.predict(X_test)
    pred2 = loaded_model.predict(X_test)
    assert np.array_equal(pred1, pred2) 