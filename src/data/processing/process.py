"""
Data processing script for the churn prediction model.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from config.mlflow_config import (
    RANDOM_SEED,
    TEST_SIZE,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_raw_data():
    """Load raw data from CSV file."""
    try:
        data = pd.read_csv("data/raw/telecom_churn.csv")
        logger.info(f"Loaded raw data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Handle missing values
        for col in NUMERICAL_FEATURES:
            data[col] = data[col].fillna(data[col].mean())
        for col in CATEGORICAL_FEATURES:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        # Handle outliers in numerical features
        for col in NUMERICAL_FEATURES:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ])

        # Split features and target
        X = data.drop(TARGET_COLUMN, axis=1)
        y = data[TARGET_COLUMN]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # Fit preprocessor on training data
        preprocessor.fit(X_train)

        # Transform data
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Save processed data
        processed_data = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor
        }

        # Save to files
        os.makedirs("data/processed", exist_ok=True)
        pd.DataFrame(X_train_processed).to_csv("data/processed/X_train.csv", index=False)
        pd.DataFrame(X_test_processed).to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        # Save original data for monitoring
        data.to_csv("data/processed/telecom_churn.csv", index=False)

        logger.info("Data preprocessing completed successfully")
        return processed_data

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def main():
    """Main data processing function."""
    try:
        # Load raw data
        data = load_raw_data()
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 