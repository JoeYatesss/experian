# Fraud Detection System

A production-ready fraud detection system with inference and training pipelines.

## Cloud Architecture (AWS)

### Training Pipeline Infrastructure
- **Data Source**: Snowflake connection via AWS PrivateLink
- **Compute**: AWS Batch for scheduled training jobs
  - Managed compute environment
  - Monthly scheduled jobs via EventBridge
- **Storage**:
  - S3 bucket for model artifacts
  - ECR for container images
- **Orchestration**: AWS Step Functions
  - Data extraction from Snowflake
  - Model training
  - Validation steps
  - Model registration

### Inference Pipeline Infrastructure
- **Compute**: ECS Fargate
  - Auto-scaling based on request load
  - Multiple availability zones
- **API Gateway**:
  - REST API endpoints
  - Request throttling
  - API key management
- **Monitoring**:
  - CloudWatch for logs and metrics
  - Model performance monitoring
  - Request/response logging
- **Security**:
  - WAF for API protection
  - Secrets Manager for credentials
  - IAM roles for service access

### Integration Points
- **Model Registry**: S3 + DynamoDB
  - S3 for model files
  - DynamoDB for metadata
- **Monitoring**: CloudWatch
  - Custom metrics for model performance
  - Automated alerts
- **CI/CD**: AWS CodePipeline
  - CodeBuild for testing
  - CodeDeploy for blue-green deployment

## Key Assumptions

### Business Requirements
1. Monthly retraining is fixed and not dynamic
2. No real-time feature engineering needed
3. All features will be provided in request (no feature store needed)
4. Single model serving (no ensemble/champion-challenger)
5. No specific latency requirements specified
6. No specific throughput requirements specified

### Technical Assumptions
1. XGBoost version compatibility between training and inference
2. JSON is acceptable for API request/response format
3. Synchronous predictions (no batch prediction needs)
4. No data retention requirements specified
5. No specific regulatory compliance requirements
6. English as primary language for logs/documentation

### Security Assumptions
1. Internal API (not public-facing)
2. API key authentication is sufficient
3. No PII in request/response data
4. No specific encryption requirements
5. Standard AWS security features are acceptable

### Operational Assumptions
1. 24/7 availability required
2. Standard monitoring tools are sufficient
3. No specific SLA requirements
4. DevOps team available for infrastructure management
5. Budget allows for managed AWS services

## System Design

### 1. Inference Pipeline
- **FastAPI Application**
  - RESTful endpoints for predictions
  - API key authentication
  - Health checks and model info endpoints
  - Real-time metrics tracking
  - Model version management

### 2. Training Pipeline Design
- **Data Processing**
  - CSV data ingestion
  - Feature preprocessing
  - Train/validation split (80/20)

- **Model Training**
  - XGBoost binary classifier
  - Hyperparameters:
    - max_depth: 3
    - learning rate: 0.3
    - objective: binary:logistic
    - eval_metric: AUC
  - Early stopping after 10 rounds
  - 100 maximum boosting rounds

- **Model Evaluation**
  - AUC-ROC score on validation set
  - Feature count tracking
  - Training sample size logging

### 3. Model Registry
- **Version Control**
  - Timestamp-based versioning (YYYYMMDD_HHMMSS)
  - Model file storage
  - Metadata tracking:
    - Version ID
    - Training metrics
    - Feature names
    - Creation timestamp

### 4. Monitoring
- **Real-time Metrics**
  - Request count
  - Error rate
  - Prediction latency
  - Prediction value distribution
  - Automatic metric reset capability

### 5. CI/CD Pipeline
- **Testing**: Automated unit tests
- **Training**: Model retraining on main branch updates
- **Deployment**: Docker container build and push
- **Artifacts**: Model storage and version tracking

## Pipeline Integration Plan

### 1. Model Lifecycle
- Training pipeline produces versioned models in `models/` directory
- Model Registry monitors directory for new versions
- Inference pipeline automatically loads latest version
- Rollback capability through version management API

### 2. Data Flow
- Training pipeline reads from data source
- Inference pipeline uses same feature preprocessing
- Monitoring data collected for retraining decisions
- Feature names and preprocessing validated between pipelines

### 3. Deployment Strategy
- Models deployed through CI/CD pipeline
- Blue-green deployment for zero-downtime updates
- Automatic rollback on performance degradation
- Model performance monitoring in production

### 4. Quality Assurance
- Model validation before deployment
- A/B testing capability for new models
- Performance metrics comparison
- Automated testing in CI/CD pipeline

## API Endpoints

- `POST /predict`: Make fraud predictions
- `GET /health`: Check system health
- `GET /metrics`: View current metrics
- `GET /model/versions`: List available models
- `GET /model/info/{version}`: Get model details

## Security

- API key authentication required for all endpoints
- Environment variable configuration
- Secure model loading and validation


Model Structure:
It has 100 decision trees (quite a substantial model)
Each tree makes decisions based on the 4 input features
Feature Importance (from most to least important):
total_balance: 1263.0 (most important feature)
total_credit_limit: 719.0
number_of_open_accounts: 479.0
number_of_accounts_in_arrears: 174.0 (least important feature)
Decision Making Process (looking at the first tree):
First checks if number_of_accounts_in_arrears < 1
Then looks at total_balance < 13451.87
Then considers total_credit_limit < 1500