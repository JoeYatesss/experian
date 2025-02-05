#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to format JSON using Python
format_json() {
    python3 -c "import json,sys;print(json.dumps(json.loads(sys.stdin.read()), indent=2))"
}

echo -e "${BLUE}1. Checking API Health${NC}"
curl -s -H "X-API-Key: default_key" http://localhost:8000/health | format_json

echo -e "\n${BLUE}2. Getting Model Info${NC}"
curl -s -H "X-API-Key: default_key" http://localhost:8000/model-info | format_json

echo -e "\n${BLUE}3. Making a Prediction${NC}"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: default_key" \
  -d '{
    "features": {
      "number_of_open_accounts": 5,
      "total_credit_limit": 50000,
      "total_balance": 25000,
      "number_of_accounts_in_arrears": 1
    }
  }' | format_json

echo -e "\n${BLUE}4. Getting Metrics${NC}"
curl -s -H "X-API-Key: default_key" http://localhost:8000/metrics | format_json 