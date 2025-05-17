import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_test_data():
    """Load or create test data."""
    # Create sample test data
    test_data = pd.DataFrame({
        'tenure': [24, 12, 36, 6, 48],
        'monthly_charges': [65.5, 45.0, 85.0, 30.0, 95.0],
        'total_charges': [1572.0, 540.0, 3060.0, 180.0, 4560.0],
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
        'partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
        'phone_service': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'multiple_lines': ['No', 'Yes', 'No', 'No', 'Yes'],
        'internet_service': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
        'online_security': ['Yes', 'No', 'Yes', 'No', 'No'],
        'online_backup': ['No', 'Yes', 'No', 'No', 'Yes'],
        'device_protection': ['Yes', 'No', 'Yes', 'No', 'No'],
        'tech_support': ['No', 'Yes', 'No', 'No', 'Yes'],
        'streaming_tv': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'streaming_movies': ['No', 'Yes', 'No', 'No', 'Yes'],
        'contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'paperless_billing': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Mailed check']
    })
    
    # Create sample labels (you would normally have real labels)
    test_labels = np.array([1, 0, 0, 1, 0])  # 1 for churn, 0 for no churn
    
    return test_data, test_labels

def test_model():
    """Test the trained model with sample data."""
    # Load the model
    model_path = os.path.join('models', 'churn_model.joblib')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model = joblib.load(model_path)
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print results
    print("\nTest Results:")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nPrediction Probabilities:")
    for i, prob in enumerate(y_pred_proba):
        print(f"Customer {i+1}: {prob:.2%} chance of churning")
    
    # Print feature importance if available
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importance = classifier.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            
            print("\nTop 5 Most Important Features:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    test_model() 