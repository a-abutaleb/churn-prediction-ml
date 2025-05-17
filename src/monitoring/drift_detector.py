import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, categorical_columns: List[str], 
                 numeric_columns: List[str], drift_threshold: float = 0.05,
                 min_window_size: int = 100):
        """
        Initialize the drift detector with reference data.
        
        Args:
            reference_data: DataFrame containing the reference data
            categorical_columns: List of categorical column names
            numeric_columns: List of numeric column names
            drift_threshold: P-value threshold for drift detection (default: 0.05)
            min_window_size: Minimum number of samples before performing drift detection
        """
        self.reference_data = reference_data
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.drift_threshold = drift_threshold
        self.min_window_size = min_window_size
        self.drift_history = []
        self.current_window = pd.DataFrame(columns=reference_data.columns)
        
        # Create monitoring directory if it doesn't exist
        os.makedirs('monitoring', exist_ok=True)
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Detect drift in the current data compared to reference data.
        
        Args:
            current_data: DataFrame containing the current data to check for drift
            
        Returns:
            Dictionary containing drift statistics for each feature
        """
        # Add current data to window
        self.current_window = pd.concat([self.current_window, current_data], ignore_index=True)
        
        # If window is too small, return no drift
        if len(self.current_window) < self.min_window_size:
            logger.info(f"Current window size ({len(self.current_window)}) is below minimum ({self.min_window_size}). Skipping drift detection.")
            return {col: {'p_value': 1.0, 'is_drift': False} for col in self.categorical_columns + self.numeric_columns}
        
        drift_results = {}
        
        # Check categorical features using chi-square test
        for col in self.categorical_columns:
            if col in current_data.columns:
                drift_results[col] = self._check_categorical_drift(col)
        
        # Check numeric features using Kolmogorov-Smirnov test
        for col in self.numeric_columns:
            if col in current_data.columns:
                drift_results[col] = self._check_numeric_drift(col)
        
        # Log drift results
        self._log_drift_results(drift_results)
        
        # Reset window if it's too large
        if len(self.current_window) >= self.min_window_size * 2:
            self.current_window = self.current_window.tail(self.min_window_size)
        
        return drift_results
    
    def _check_categorical_drift(self, column: str) -> Dict[str, float]:
        """Check drift in categorical features using chi-square test."""
        ref_counts = self.reference_data[column].value_counts()
        curr_counts = self.current_window[column].value_counts()
        
        # Align the counts
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_counts = ref_counts.reindex(all_categories, fill_value=0)
        curr_counts = curr_counts.reindex(all_categories, fill_value=0)
        
        # Scale reference counts to match current window size
        scale_factor = len(self.current_window) / len(self.reference_data)
        ref_counts = ref_counts * scale_factor
        
        try:
            chi2, p_value = stats.chisquare(curr_counts, ref_counts)
            return {
                'p_value': float(p_value),
                'is_drift': p_value < self.drift_threshold,
                'chi2_statistic': float(chi2)
            }
        except Exception as e:
            logger.warning(f"Error in chi-square test for {column}: {str(e)}")
            return {
                'p_value': 1.0,
                'is_drift': False,
                'chi2_statistic': 0.0
            }
    
    def _check_numeric_drift(self, column: str) -> Dict[str, float]:
        """Check drift in numeric features using Kolmogorov-Smirnov test."""
        ref_data = self.reference_data[column].dropna()
        curr_data = self.current_window[column].dropna()
        
        try:
            ks_statistic, p_value = stats.ks_2samp(ref_data, curr_data)
            return {
                'p_value': float(p_value),
                'is_drift': p_value < self.drift_threshold,
                'ks_statistic': float(ks_statistic)
            }
        except Exception as e:
            logger.warning(f"Error in KS test for {column}: {str(e)}")
            return {
                'p_value': 1.0,
                'is_drift': False,
                'ks_statistic': 0.0
            }
    
    def _log_drift_results(self, drift_results: Dict[str, Dict[str, float]]) -> None:
        """Log drift detection results to a file."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'drift_results': drift_results,
            'window_size': len(self.current_window)
        }
        
        self.drift_history.append(log_entry)
        
        # Save to file
        log_file = 'monitoring/drift_history.json'
        with open(log_file, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
        
        # Log summary
        drifted_features = [col for col, results in drift_results.items() 
                          if results.get('is_drift', False)]
        
        if drifted_features:
            logger.warning(f"Drift detected in features: {drifted_features}")
        else:
            logger.info("No drift detected in any features")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get a summary of drift detection history."""
        if not self.drift_history:
            return {"message": "No drift detection history available"}
        
        latest_results = self.drift_history[-1]['drift_results']
        drifted_features = [col for col, results in latest_results.items() 
                          if results.get('is_drift', False)]
        
        return {
            'timestamp': self.drift_history[-1]['timestamp'],
            'drifted_features': drifted_features,
            'total_features_checked': len(latest_results),
            'drift_threshold': self.drift_threshold,
            'current_window_size': len(self.current_window)
        } 