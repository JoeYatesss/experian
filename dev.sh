#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        exit 1
    fi
}

# Function to build and start services
start_services() {
    echo -e "${BLUE}Building and starting services...${NC}"
    docker-compose build
    docker-compose up -d api
    echo -e "${GREEN}Services started!${NC}"
}

# Function to run tests
run_tests() {
    echo -e "${BLUE}Running tests...${NC}"
    docker-compose run --rm test
}

# Function to show API logs
show_logs() {
    echo -e "${BLUE}Showing API logs...${NC}"
    docker-compose logs -f api
}

# Function to run the test script
run_test_script() {
    echo -e "${BLUE}Running test script...${NC}"
    docker-compose exec api ./test_api.sh
}

# Main script
case "$1" in
    "start")
        check_docker
        start_services
        ;;
    "test")
        check_docker
        run_tests
        ;;
    "logs")
        check_docker
        show_logs
        ;;
    "test-api")
        check_docker
        run_test_script
        ;;
    *)
        echo "Usage: $0 {start|test|logs|test-api}"
        echo "  start     - Build and start the API service"
        echo "  test      - Run the test suite"
        echo "  logs      - Show API logs"
        echo "  test-api  - Run the API test script"
        exit 1
        ;;
esac 