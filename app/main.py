from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
import xgboost as xgb
from typing import Dict
import os
import logging
from contextlib import asynccontextmanager
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "default_key")
API_VERSION = "1.0.0"

class Metrics:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.prediction_times = []
        self.prediction_values = []
        self.last_metrics_reset = datetime.now()

    def reset(self):
        self.__init__()

metrics = Metrics()

api_key_header = APIKeyHeader(name="X-API-Key")

# loading the model
model = None
feature_names = [
    "number_of_open_accounts",
    "total_credit_limit",
    "total_balance",
    "number_of_accounts_in_arrears"
]

def load_model():
    """Load the model from xgboost.json"""
    global model
    try:
        model_path = "xgboost.json"
        if not os.path.exists(model_path):
            logger.error("Model file not found")
            return False
            
        model = xgb.Booster()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def load_for_test():
    """Load model for testing"""
    if not load_model():
        logger.error("Failed to load model for testing")
    return model is not None

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not load_model():
        logger.error("Failed to load model during startup")
    yield
    # Shutdown: Clean up resources if needed
    global model
    model = None
    logger.info("Cleaned up model resources")

app = FastAPI(
    title="Fraud Detection API",
    lifespan=lifespan
)
app.state.MODEL_PATH = "xgboost.json"  # Default model path

# Pydantic models for request/response validation
class Features(BaseModel):
    number_of_open_accounts: int = Field(..., description="Number of open accounts")
    total_credit_limit: int = Field(..., description="Total credit limit")
    total_balance: float = Field(..., description="Total balance")
    number_of_accounts_in_arrears: int = Field(..., description="Number of accounts in arrears")

class PredictionRequest(BaseModel):
    features: Features

class PredictionResponse(BaseModel):
    fraud_probability: float
    version: str
    risk_level: str = Field(..., description="Risk level classification")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    confidence_metrics: Dict[str, float] = Field(..., description="Additional confidence metrics")

# Security dependency
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.get("/health")
async def health_check(_: str = Depends(verify_api_key)):
    """Check if the API and model are healthy"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": API_VERSION
    }

@app.get("/model-info")
async def model_info(_: str = Depends(verify_api_key)):
    """Get information about the model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "version": API_VERSION,
        "model_type": "XGBoost",
        "features": feature_names
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    _: str = Depends(verify_api_key)
):
    """Make a fraud prediction"""
    if model is None:
        metrics.error_count += 1
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        start_time = time.time()
        metrics.request_count += 1

        # Convert features to list in the correct order
        feature_values = []
        for feature in feature_names:
            feature_values.append(getattr(request.features, feature))
        
        # Create DMatrix with feature names
        dmatrix = xgb.DMatrix(
            [feature_values],
            feature_names=feature_names
        )
        
        # Make prediction
        prediction = model.predict(dmatrix)
        prediction_value = float(prediction[0])
        
        # Calculate risk level
        risk_level = "HIGH" if prediction_value > 0.7 else "MEDIUM" if prediction_value > 0.3 else "LOW"
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time())}_{hash(str(feature_values))}"[:16]
        
        # Update metrics
        prediction_time = time.time() - start_time
        metrics.prediction_times.append(prediction_time)
        metrics.prediction_values.append(prediction_value)
        
        # Calculate confidence metrics
        confidence_metrics = {
            "prediction_time_ms": round(prediction_time * 1000, 2),
            "model_confidence": round(abs(prediction_value - 0.5) * 2, 2),  # Higher for more extreme predictions
            "feature_completeness": 1.0  # All features are required
        }
        
        return {
            "fraud_probability": prediction_value,
            "version": API_VERSION,
            "risk_level": risk_level,
            "prediction_timestamp": datetime.now().isoformat(),
            "prediction_id": prediction_id,
            "confidence_metrics": confidence_metrics
        }
        
    except Exception as e:
        metrics.error_count += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(_: str = Depends(verify_api_key)):
    """Get current metrics"""
    if not metrics.prediction_times:
        return {
            "status": "no_predictions_yet",
            "request_count": 0,
            "error_count": 0
        }
    
    return {
        "request_count": metrics.request_count,
        "error_count": metrics.error_count,
        "avg_prediction_time": sum(metrics.prediction_times) / len(metrics.prediction_times),
        "avg_prediction_value": sum(metrics.prediction_values) / len(metrics.prediction_values),
        "total_predictions": len(metrics.prediction_values),
        "since": metrics.last_metrics_reset.isoformat()
    }

@app.post("/metrics/reset")
async def reset_metrics(_: str = Depends(verify_api_key)):
    """Reset all metrics"""
    metrics.reset()
    return {"status": "metrics_reset"} 