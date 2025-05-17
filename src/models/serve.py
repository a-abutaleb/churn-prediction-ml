import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import uvicorn
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
loaded_model = None
model_info = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    try:
        # Get the latest model from MLflow
        mlflow.set_tracking_uri("http://localhost:5050")
        client = mlflow.tracking.MlflowClient()
        
        # Try to get the latest production model
        try:
            global model_info
            model_info = client.get_latest_versions("churn_prediction_model", stages=["Production"])[0]
            model_uri = f"models:/churn_prediction_model/Production"
            
            # Load the model
            global loaded_model
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model version {model_info.version} from MLflow")
        except Exception as e:
            logger.warning(f"Could not load production model: {str(e)}")
            # Try to load latest model from any stage
            try:
                registered_model = client.search_registered_models(filter_string="name='churn_prediction_model'")[0]
                latest_version = registered_model.latest_versions[0]
                model_uri = f"models:/churn_prediction_model/{latest_version.version}"
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                model_info = latest_version
                logger.info(f"Loaded latest available model version {latest_version.version}")
            except Exception as e:
                logger.error(f"Could not load any model: {str(e)}")
                loaded_model = None
                model_info = None
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    
    yield
    
    # Cleanup
    global loaded_model, model_info
    loaded_model = None
    model_info = None

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using MLflow deployed model",
    version="1.0.0",
    lifespan=lifespan
)

# Define input schema
class ChurnPredictionInput(BaseModel):
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    monthly_charges: float = Field(..., description="The monthly charges of the customer")
    total_charges: float = Field(..., description="The total charges of the customer")
    gender: str = Field(..., description="Customer's gender (Male/Female)")
    partner: str = Field(..., description="Whether the customer has a partner (Yes/No)")
    dependents: str = Field(..., description="Whether the customer has dependents (Yes/No)")
    phone_service: str = Field(..., description="Whether the customer has phone service (Yes/No)")
    multiple_lines: str = Field(..., description="Whether the customer has multiple lines")
    internet_service: str = Field(..., description="Type of internet service")
    online_security: str = Field(..., description="Whether the customer has online security")
    online_backup: str = Field(..., description="Whether the customer has online backup")
    device_protection: str = Field(..., description="Whether the customer has device protection")
    tech_support: str = Field(..., description="Whether the customer has tech support")
    streaming_tv: str = Field(..., description="Whether the customer has streaming TV")
    streaming_movies: str = Field(..., description="Whether the customer has streaming movies")
    contract: str = Field(..., description="The contract term")
    paperless_billing: str = Field(..., description="Whether the customer has paperless billing")
    payment_method: str = Field(..., description="The customer's payment method")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 80.5,
                "total_charges": 966.0,
                "gender": "Male",
                "partner": "Yes",
                "dependents": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": loaded_model is not None
    }

@app.post("/predict")
async def predict(input_data: ChurnPredictionInput):
    """Make predictions using the loaded model"""
    if loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the model to be loaded or contact the administrator."
        )
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.model_dump()
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = loaded_model.predict(df)
        probability = loaded_model.predict_proba(df)[0][1]  # Probability of churn
        
        return {
            "churn_prediction": bool(prediction[0]),
            "churn_probability": float(probability),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
async def get_model_metadata():
    """Get metadata about the currently loaded model"""
    if model_info is None:
        return {
            "status": "No model loaded",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "model_name": "churn_prediction_model",
        "model_version": model_info.version,
        "creation_timestamp": model_info.creation_timestamp,
        "current_stage": model_info.current_stage,
        "description": model_info.description,
        "status": "loaded",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5001, reload=True) 