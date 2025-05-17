import mlflow
from mlflow.tracking import MlflowClient
import os
from datetime import datetime

class ModelRegistry:
    def __init__(self, model_name="wine_quality_model"):
        self.client = MlflowClient()
        self.model_name = model_name
        
    def register_model(self, run_id, model_path="model"):
        """Register a model from a specific run"""
        try:
            # Register the model
            model_uri = f"runs:/{run_id}/{model_path}"
            model_details = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id
            )
            print(f"Model version {model_details.version} registered")
            return model_details
        except Exception as e:
            print(f"Error registering model: {e}")
            raise
    
    def transition_model_stage(self, version, stage):
        """Transition a model version to a specific stage"""
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            print(f"Model version {version} transitioned to {stage}")
        except Exception as e:
            print(f"Error transitioning model stage: {e}")
            raise
    
    def get_latest_model(self, stage=None):
        """Get the latest model version for a specific stage"""
        try:
            if stage:
                model_details = self.client.get_latest_versions(
                    self.model_name,
                    stages=[stage]
                )[0]
            else:
                model_details = self.client.get_latest_versions(
                    self.model_name
                )[0]
            return model_details
        except Exception as e:
            print(f"Error getting latest model: {e}")
            raise
    
    def get_model_metadata(self, version):
        """Get metadata for a specific model version"""
        try:
            model_details = self.client.get_model_version(
                self.model_name,
                version
            )
            return {
                "version": model_details.version,
                "stage": model_details.current_stage,
                "run_id": model_details.run_id,
                "status": model_details.status,
                "created_at": datetime.fromtimestamp(model_details.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"Error getting model metadata: {e}")
            raise
    
    def list_model_versions(self):
        """List all versions of the model"""
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return [{
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status
            } for v in versions]
        except Exception as e:
            print(f"Error listing model versions: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Register a model
    run_id = "example-run-id"
    model_details = registry.register_model(run_id)
    
    # Transition to staging
    registry.transition_model_stage(model_details.version, "Staging")
    
    # Get latest staging model
    staging_model = registry.get_latest_model("Staging")
    print(f"Latest staging model: {staging_model.version}")
    
    # Get model metadata
    metadata = registry.get_model_metadata(staging_model.version)
    print(f"Model metadata: {metadata}")
    
    # List all versions
    versions = registry.list_model_versions()
    print(f"All model versions: {versions}") 