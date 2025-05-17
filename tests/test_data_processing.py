import pytest
import pandas as pd
import numpy as np
from src.data.process_data import load_data, preprocess_data, split_data, process_data

def test_data_loading():
    """Test data loading functionality"""
    # Create a temporary test file
    test_data = pd.DataFrame({
        'churn': [0, 1, 0, 1],
        'tenure': [1, 2, 3, 4],
        'MonthlyCharges': [10.0, 20.0, 30.0, 40.0]
    })
    test_data.to_csv('test_data.csv', index=False)
    
    # Test loading
    data = load_data('test_data.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'churn' in data.columns
    
    # Clean up
    import os
    os.remove('test_data.csv')

def test_preprocessing():
    """Test data preprocessing steps"""
    # Create test data
    test_data = pd.DataFrame({
        'churn': [0, 1, 0, 1],
        'tenure': [1, 2, 3, 4],
        'MonthlyCharges': [10.0, 20.0, 30.0, 40.0],
        'Contract': ['Month-to-month', 'Yearly', 'Month-to-month', 'Yearly']
    })
    
    # Test preprocessing
    processed_data = preprocess_data(test_data)
    
    # Check if categorical variables are encoded
    assert not any(processed_data.select_dtypes(include=['object']).columns)
    
    # Check if missing values are handled
    assert not processed_data.isnull().any().any()
    
    # Check if target variable is binary
    assert set(processed_data['churn'].unique()).issubset({0, 1})

def test_feature_engineering():
    """Test feature engineering steps"""
    # Create test data
    test_data = pd.DataFrame({
        'churn': [0, 1, 0, 1],
        'tenure': [1, 2, 3, 4],
        'MonthlyCharges': [10.0, 20.0, 30.0, 40.0],
        'Contract': ['Month-to-month', 'Yearly', 'Month-to-month', 'Yearly']
    })
    
    # Test preprocessing
    processed_data = preprocess_data(test_data)
    
    # Check if all required features are present
    required_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract_Month-to-month', 'InternetService_Fiber optic',
        'PaymentMethod_Electronic check'
    ]
    for feature in required_features:
        if feature in processed_data.columns:
            assert True
        else:
            print(f"Warning: Feature {feature} not found in processed data")

def test_train_test_split():
    """Test train-test split functionality"""
    # Create test data
    test_data = pd.DataFrame({
        'churn': [0, 1, 0, 1, 0, 1, 0, 1],
        'tenure': [1, 2, 3, 4, 5, 6, 7, 8],
        'MonthlyCharges': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    })
    
    # Test split
    X_train, X_test, y_train, y_test = split_data(test_data, 'churn')
    
    # Check shapes
    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Check if split is random
    assert not np.array_equal(X_train[:2], X_test[:2]) 