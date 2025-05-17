import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
import mlflow
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data, model, threshold=0.05):
        self.reference_data = reference_data
        self.model = model
        self.threshold = threshold
        self.reference_stats = self._calculate_reference_stats()
    
    def _calculate_reference_stats(self):
        """Calculate statistics for reference data"""
        stats = {}
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['float64', 'int64']:
                stats[column] = {
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'ks_statistic': None  # Will be calculated during drift detection
                }
        return stats
    
    def detect_data_drift(self, new_data):
        """Detect data drift using Kolmogorov-Smirnov test"""
        drift_results = {}
        for column in self.reference_stats.keys():
            if column in new_data.columns:
                # Perform KS test
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[column],
                    new_data[column]
                )
                
                # Update reference statistics
                self.reference_stats[column]['ks_statistic'] = ks_statistic
                
                # Check for drift
                is_drift = p_value < self.threshold
                drift_results[column] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'is_drift': is_drift
                }
                
                if is_drift:
                    logger.warning(f"Data drift detected in column {column}")
        
        return drift_results
    
    def detect_model_drift(self, new_data, new_labels):
        """Detect model drift by comparing performance"""
        # Get reference predictions
        reference_predictions = self.model.predict(self.reference_data)
        reference_accuracy = accuracy_score(
            self.reference_data['quality'],
            reference_predictions
        )
        
        # Get new predictions
        new_predictions = self.model.predict(new_data)
        new_accuracy = accuracy_score(new_labels, new_predictions)
        
        # Calculate performance difference
        performance_diff = abs(reference_accuracy - new_accuracy)
        is_drift = performance_diff > self.threshold
        
        drift_result = {
            'reference_accuracy': reference_accuracy,
            'new_accuracy': new_accuracy,
            'performance_diff': performance_diff,
            'is_drift': is_drift
        }
        
        if is_drift:
            logger.warning("Model drift detected")
        
        return drift_result
    
    def log_drift_metrics(self, drift_results, drift_type="data"):
        """Log drift detection metrics to MLflow"""
        with mlflow.start_run(nested=True):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mlflow.log_param("drift_type", drift_type)
            mlflow.log_param("timestamp", timestamp)
            
            if drift_type == "data":
                for column, results in drift_results.items():
                    mlflow.log_metric(f"{column}_ks_statistic", results['ks_statistic'])
                    mlflow.log_metric(f"{column}_p_value", results['p_value'])
                    mlflow.log_metric(f"{column}_is_drift", int(results['is_drift']))
            else:
                mlflow.log_metric("reference_accuracy", drift_results['reference_accuracy'])
                mlflow.log_metric("new_accuracy", drift_results['new_accuracy'])
                mlflow.log_metric("performance_diff", drift_results['performance_diff'])
                mlflow.log_metric("is_drift", int(drift_results['is_drift']))

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    
    # Load data
    data = pd.read_csv('data/winequality-red.csv')
    reference_data = data.sample(frac=0.7, random_state=42)
    new_data = data.drop(reference_data.index)
    
    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(reference_data.drop('quality', axis=1), reference_data['quality'])
    
    # Initialize drift detector
    detector = DriftDetector(reference_data, model)
    
    # Detect data drift
    data_drift = detector.detect_data_drift(new_data)
    detector.log_drift_metrics(data_drift, "data")
    
    # Detect model drift
    model_drift = detector.detect_model_drift(
        new_data.drop('quality', axis=1),
        new_data['quality']
    )
    detector.log_drift_metrics(model_drift, "model") 