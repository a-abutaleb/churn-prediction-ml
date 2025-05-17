import os
import mlflow
import pandas as pd
import numpy as np
import optuna
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Dict, Tuple, Any, List
from mlflow.models import infer_signature

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> None:
    """Validate the input data."""
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        raise ValueError(f"Found missing values:\n{missing_values[missing_values > 0]}")
    
    # Check data types
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} should be numeric but found {df[col].dtype}")

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the telecom churn dataset."""
    try:
        # Load data
        df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Convert TotalCharges to numeric, replacing empty strings with NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill NaN values in TotalCharges with 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Validate data
        validate_data(df)
        
        # Separate features and target
        X = df.drop(['Churn', 'customerID'], axis=1)
        y = (df['Churn'] == 'Yes').astype(int)  # Convert to binary
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(X: pd.DataFrame) -> ColumnTransformer:
    """Preprocess the features."""
    # Define numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor

def objective_xgboost(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> float:
    """Optuna objective function for XGBoost hyperparameter tuning."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }
    
    model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            **params
        ))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced n_splits
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

def objective_rf(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> float:
    """Optuna objective function for Random Forest hyperparameter tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            random_state=42,
            **params
        ))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced n_splits
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

def objective_lr(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> float:
    """Optuna objective function for Logistic Regression hyperparameter tuning."""
    params = {
        'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'max_iter': trial.suggest_int('max_iter', 100, 1000)
    }
    
    model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            random_state=42,
            **params
        ))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced n_splits
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

def train_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> List[Tuple[str, Pipeline, Dict[str, Any]]]:
    """Train multiple models with hyperparameter tuning."""
    try:
        models = []
        
        # Train XGBoost
        logger.info("Tuning XGBoost hyperparameters...")
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda trial: objective_xgboost(trial, X_train, y_train, preprocessor), n_trials=20)
        
        xgb_model = ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                **study_xgb.best_params
            ))
        ])
        xgb_model.fit(X_train, y_train)
        models.append(('xgboost', xgb_model, study_xgb.best_params))
        
        # Train Random Forest
        logger.info("Tuning Random Forest hyperparameters...")
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, preprocessor), n_trials=20)
        
        rf_model = ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(
                random_state=42,
                **study_rf.best_params
            ))
        ])
        rf_model.fit(X_train, y_train)
        models.append(('random_forest', rf_model, study_rf.best_params))
        
        # Train Logistic Regression
        logger.info("Tuning Logistic Regression hyperparameters...")
        study_lr = optuna.create_study(direction='maximize')
        study_lr.optimize(lambda trial: objective_lr(trial, X_train, y_train, preprocessor), n_trials=20)
        
        lr_model = ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(
                random_state=42,
                **study_lr.best_params
            ))
        ])
        lr_model.fit(X_train, y_train)
        models.append(('logistic_regression', lr_model, study_lr.best_params))
        
        return models
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def evaluate_model(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_test: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate the model performance."""
    try:
        # Cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std())
        }
        
        # Get feature importance if available
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            feature_importance = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            feature_importance = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            feature_importance = None
            
        if feature_importance is not None:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importance_dict = {str(name): float(imp) for name, imp in zip(feature_names, feature_importance)}
        else:
            importance_dict = {}
        
        return metrics, importance_dict
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    try:
        # Set MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5050")
        mlflow.set_experiment("telecom_churn_prediction")
        
        logger.info("Loading data...")
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Preprocessing data...")
        preprocessor = preprocess_data(X_train)
        
        logger.info("Training models...")
        models = train_models(X_train, y_train, preprocessor)
        
        best_score = -np.inf
        best_model_name = None
        best_model = None
        
        for model_name, model, params in models:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                logger.info(f"Evaluating {model_name}...")
                metrics, feature_importance = evaluate_model(model, X_train, X_test, y_train, y_test)
                
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log feature importance
                if feature_importance:
                    mlflow.log_dict(feature_importance, "feature_importance.json")
                
                # Create model signature
                signature = infer_signature(X_train, model.predict(X_train))
                
                # Log model with signature
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=f"churn_prediction_{model_name}",
                    signature=signature,
                    input_example=X_train.iloc[:1]
                )
                
                logger.info(f"\n{model_name} metrics:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"{metric_name}: {metric_value:.4f}")
                
                if feature_importance:
                    logger.info(f"\nTop 5 Important Features for {model_name}:")
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    for feature, importance in sorted_features[:5]:
                        logger.info(f"{feature}: {importance:.4f}")
                
                # Track best model
                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    best_model_name = model_name
                    best_model = model
        
        # Save best model locally
        logger.info(f"\nSaving best model ({best_model_name})...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/churn_model.joblib')
        
        logger.info(f"\nBest model: {best_model_name} with ROC AUC: {best_score:.4f}")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 