import os
import sys
import logging
import pandas as pd
import kaggle
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_kaggle_credentials():
    """Set up Kaggle credentials if not already configured."""
    try:
        # Check if kaggle.json exists
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
            
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        if not os.path.exists(kaggle_json):
            logger.error("""
            Please set up your Kaggle credentials first:
            1. Go to https://www.kaggle.com/account
            2. Click on 'Create New API Token'
            3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
            4. Run: chmod 600 ~/.kaggle/kaggle.json
            """)
            sys.exit(1)
            
        # Set permissions
        os.chmod(kaggle_json, 0o600)
        
    except Exception as e:
        logger.error(f"Error setting up Kaggle credentials: {str(e)}")
        sys.exit(1)

def download_dataset():
    """Download the Telco Customer Churn dataset from Kaggle."""
    try:
        # Create data directories if they don't exist
        data_dir = Path('data')
        raw_dir = data_dir / 'raw'
        processed_dir = data_dir / 'processed'
        
        for dir_path in [data_dir, raw_dir, processed_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Download dataset
        logger.info("Downloading Telco Customer Churn dataset...")
        kaggle.api.dataset_download_files(
            'blastchar/telco-customer-churn',
            path=raw_dir,
            unzip=True
        )
        
        # Validate downloaded data
        dataset_path = raw_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        if not dataset_path.exists():
            raise FileNotFoundError("Dataset file not found after download")
            
        # Load and validate data
        df = pd.read_csv(dataset_path)
        validate_dataset(df)
        
        # Save processed dataset
        logger.info("Processing and saving dataset...")
        process_dataset(df).to_csv(processed_dir / 'telco_churn.csv', index=False)
        logger.info("Dataset downloaded and processed successfully!")
        
        # Log dataset info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        sys.exit(1)

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the downloaded dataset."""
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
    
    # Check for minimum number of rows
    if len(df) < 1000:
        raise ValueError("Dataset is too small, expected at least 1000 rows")
    
    # Check for target variable distribution
    churn_dist = df['Churn'].value_counts(normalize=True)
    if churn_dist.min() < 0.1:
        logger.warning("Severe class imbalance detected in target variable")

def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataset for modeling."""
    df_processed = df.copy()
    
    # Convert TotalCharges to numeric
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'].str.strip(), errors='coerce')
    
    # Fill missing values
    df_processed['TotalCharges'].fillna(0, inplace=True)
    
    # Convert binary variables
    df_processed['Churn'] = (df_processed['Churn'] == 'Yes').astype(int)
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
    
    # Convert yes/no columns to binary
    binary_columns = ['PhoneService', 'PaperlessBilling', 'Partner', 'Dependents']
    for col in binary_columns:
        df_processed[col] = (df_processed[col] == 'Yes').astype(int)
    
    return df_processed

def main():
    """Main function to download and prepare the dataset."""
    logger.info("Starting dataset download process...")
    setup_kaggle_credentials()
    download_dataset()

if __name__ == "__main__":
    main() 