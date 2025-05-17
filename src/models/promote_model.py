import mlflow
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME
)

def promote_to_production():
    """Promote the latest model version to Production stage"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    
    # Get the latest version
    latest_version = max([int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'")])
    
    # Transition the model to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production"
    )
    
    print(f"Model {MODEL_NAME} version {latest_version} promoted to Production")

if __name__ == "__main__":
    promote_to_production() 