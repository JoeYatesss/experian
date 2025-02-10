from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import xgboost as xgb
import os
from datetime import datetime

# Basic setup
app = FastAPI(title="Fraud Detection API")
API_KEY = os.getenv("API_KEY", "default_key")
api_key_header = APIKeyHeader(name="X-API-Key")
model = None

# Load model
def load_model(model_path=None):
    global model
    try:
        model = xgb.Booster()
        model_path = model_path or "xgboost.json"
        model.load_model(model_path)
        # Set model parameters to match training
        model.set_param('max_depth', 6)
        model.set_param('eta', 0.3)
        model.set_param('objective', 'binary:logistic')
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Function for testing
def load_for_test():
    """Special function for testing that loads a test model"""
    return load_model(os.getenv("MODEL_PATH", "test_xgboost.json"))

# Models
class Features(BaseModel):
    number_of_open_accounts: int
    total_credit_limit: int
    total_balance: float
    number_of_accounts_in_arrears: int

class PredictionRequest(BaseModel):
    features: Features

# Security
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Routes
@app.get("/health")
async def health_check(_: str = Depends(verify_api_key)):
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/model-info")
async def model_info(_: str = Depends(verify_api_key)):
    return {
        "version": "1.0.0",
        "model_type": "XGBoost",
        "features": [
            "number_of_open_accounts",
            "total_credit_limit",
            "total_balance",
            "number_of_accounts_in_arrears"
        ]
    }

@app.post("/predict")
async def predict(request: PredictionRequest, _: str = Depends(verify_api_key)):
    if model is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare features
        features = [
            request.features.number_of_open_accounts,
            request.features.total_credit_limit,
            request.features.total_balance,
            request.features.number_of_accounts_in_arrears
        ]
        
        # Predict
        feature_names = ["number_of_open_accounts", "total_credit_limit", "total_balance", "number_of_accounts_in_arrears"]
        dmatrix = xgb.DMatrix([features], feature_names=feature_names)
        prediction = float(model.predict(dmatrix)[0])
        
        # Determine risk level
        risk_level = "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.3 else "LOW"
        
        # Calculate credit utilization
        credit_utilization = (request.features.total_balance / request.features.total_credit_limit * 100) if request.features.total_credit_limit > 0 else 0
        
        return {
            "prediction": {
                "fraud_probability": f"{prediction:.1%}",
                "risk_level": risk_level,
                "risk_factors": {
                    "credit_utilization": f"{credit_utilization:.1f}%",
                    "accounts_in_arrears": request.features.number_of_accounts_in_arrears,
                    "total_accounts": request.features.number_of_open_accounts
                }
            },
            "input_summary": {
                "total_balance": f"${request.features.total_balance:,.2f}",
                "credit_limit": f"${request.features.total_credit_limit:,.2f}",
                "open_accounts": request.features.number_of_open_accounts,
                "accounts_in_arrears": request.features.number_of_accounts_in_arrears
            },
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))