#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# API Key from environment or default
API_KEY=${API_KEY:-"test_api_key_123"}

# Function to format JSON using Python
format_json() {
    python3 -c "import json,sys;print(json.dumps(json.loads(sys.stdin.read()), indent=2))"
}

echo -e "${BLUE}1. Testing Health Check${NC}"
curl -s -H "X-API-Key: ${API_KEY}" http://localhost:8000/health | format_json
echo

echo -e "\n${BLUE}2. Testing Model Info${NC}"
curl -s -H "X-API-Key: ${API_KEY}" http://localhost:8000/model-info | format_json
echo

echo -e "\n${BLUE}3. Testing Low Risk Prediction${NC}"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "features": {
      "number_of_open_accounts": 5,
      "total_credit_limit": 50000,
      "total_balance": 10000,
      "number_of_accounts_in_arrears": 0
    }
  }' | format_json
echo

echo -e "\n${BLUE}4. Testing High Risk Prediction${NC}"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "features": {
      "number_of_open_accounts": 12,
      "total_credit_limit": 10000,
      "total_balance": 9500,
      "number_of_accounts_in_arrears": 8
    }
  }' | format_json
echo

echo -e "\n${BLUE}5. Testing Invalid Input (should fail)${NC}"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "features": {
      "number_of_open_accounts": "invalid",
      "total_credit_limit": 50000,
      "total_balance": 25000,
      "number_of_accounts_in_arrears": 0
    }
  }' | format_json
echo 