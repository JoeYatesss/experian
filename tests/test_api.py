import pytest
from fastapi.testclient import TestClient
from app.main import app, API_KEY, load_for_test
import xgboost as xgb
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set test model path
TEST_MODEL_PATH = "test_xgboost.json"
app.state.MODEL_PATH = TEST_MODEL_PATH

@pytest.fixture(scope="session", autouse=True)
def create_test_model():
    """Create a test model if it doesn't exist"""
    try:
        if not os.path.exists(TEST_MODEL_PATH):
            logger.info(f"Creating test model at {TEST_MODEL_PATH}")
            # Create sample data with correct feature names
            feature_names = [
                "number_of_open_accounts",
                "total_credit_limit",
                "total_balance",
                "number_of_accounts_in_arrears"
            ]
            X = np.random.rand(100, len(feature_names))
            y = np.random.randint(0, 2, 100)
            dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
            
            # Create and train the model
            bst = xgb.Booster()
            bst = xgb.train(
                params={'max_depth': 2, 'eta': 0.3, 'objective': 'binary:logistic'},
                dtrain=dtrain,
                num_boost_round=2
            )
            bst.save_model(TEST_MODEL_PATH)
            logger.info("Test model created successfully")
        else:
            logger.info(f"Test model already exists at {TEST_MODEL_PATH}")
        
        # Verify the model file exists and is readable
        assert os.path.exists(TEST_MODEL_PATH), f"Model file not found at {TEST_MODEL_PATH}"
        assert os.access(TEST_MODEL_PATH, os.R_OK), f"Model file not readable at {TEST_MODEL_PATH}"
        
        # Load the model for testing
        assert load_for_test(), "Failed to load model for testing"
        logger.info("Test model loaded successfully")
        
        yield TEST_MODEL_PATH
        
        # Cleanup after tests
        if os.path.exists(TEST_MODEL_PATH):
            os.remove(TEST_MODEL_PATH)
            logger.info(f"Test model cleaned up at {TEST_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error in test model setup: {str(e)}")
        raise

client = TestClient(app)

# Test data
valid_features = {
    "features": {
        "credit_score": 750,
        "transaction_amount": 1000
    }
}

invalid_features = {
    "features": {
        "credit_score": 750
        # Missing transaction_amount
    }
}

def get_headers():
    return {"X-API-Key": API_KEY}

def test_health_check():
    response = client.get("/health", headers=get_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "version" in data

def test_health_check_invalid_api_key():
    response = client.get("/health", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 403

def test_model_info():
    response = client.get("/model-info", headers=get_headers())
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "features" in data
    assert "model_type" in data
    assert data["model_type"] == "XGBoost"
    # Verify the expected features are present
    expected_features = [
        "number_of_open_accounts",
        "total_credit_limit",
        "total_balance",
        "number_of_accounts_in_arrears"
    ]
    assert all(feature in data["features"] for feature in expected_features)

def test_predict_fraud_risk():
    # Test data based on the feature names from xgboost.json
    test_input = {
        "features": {
            "number_of_open_accounts": 3,
            "total_credit_limit": 50000,
            "total_balance": 25000.0,
            "number_of_accounts_in_arrears": 0
        }
    }

    response = client.post("/predict", json=test_input, headers=get_headers())
    
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "version" in data
    assert isinstance(data["fraud_probability"], float)
    assert 0 <= data["fraud_probability"] <= 1

def test_predict_invalid_input():
    # Test with missing fields
    invalid_input = {
        "features": {
            "number_of_open_accounts": 3,
            "total_credit_limit": 50000
            # Missing total_balance and number_of_accounts_in_arrears
        }
    }

    response = client.post("/predict", json=invalid_input, headers=get_headers())
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_invalid_values():
    # Test with invalid data types
    invalid_input = {
        "features": {
            "number_of_open_accounts": "invalid",  # Should be int
            "total_credit_limit": 50000,
            "total_balance": 25000.0,
            "number_of_accounts_in_arrears": 0
        }
    }

    response = client.post("/predict", json=invalid_input, headers=get_headers())
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_missing_api_key():
    test_input = {
        "features": {
            "number_of_open_accounts": 3,
            "total_credit_limit": 50000,
            "total_balance": 25000.0,
            "number_of_accounts_in_arrears": 0
        }
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 403

def test_predict_invalid_api_key():
    test_input = {
        "features": {
            "number_of_open_accounts": 3,
            "total_credit_limit": 50000,
            "total_balance": 25000.0,
            "number_of_accounts_in_arrears": 0
        }
    }
    response = client.post("/predict", json=test_input, headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 403 