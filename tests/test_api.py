import pytest
from fastapi.testclient import TestClient
from app.main import app, API_KEY, load_for_test
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    try:
        # Load the model for testing
        assert load_for_test(), "Failed to load model for testing"
        logger.info("Model loaded successfully for testing")
        yield
    except Exception as e:
        logger.error(f"Error in test setup: {str(e)}")
        raise

client = TestClient(app)

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
    expected_features = [
        "number_of_open_accounts",
        "total_credit_limit",
        "total_balance",
        "number_of_accounts_in_arrears"
    ]
    assert all(feature in data["features"] for feature in expected_features)

def test_predict_fraud_risk():
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
    
    # Check main structure
    assert "prediction" in data
    assert "input_summary" in data
    assert "timestamp" in data
    assert "version" in data
    
    # Check prediction details
    prediction = data["prediction"]
    assert "fraud_probability" in prediction
    assert "risk_level" in prediction
    assert "risk_factors" in prediction
    
    # Check risk factors
    risk_factors = prediction["risk_factors"]
    assert "credit_utilization" in risk_factors
    assert "accounts_in_arrears" in risk_factors
    assert "total_accounts" in risk_factors
    
    # Check input summary
    input_summary = data["input_summary"]
    assert "total_balance" in input_summary
    assert "credit_limit" in input_summary
    assert "open_accounts" in input_summary
    assert "accounts_in_arrears" in input_summary

def test_predict_invalid_input():
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