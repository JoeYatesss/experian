# Fraud Detection System

## Quick Start for Development

### Prerequisites
- Python 3.9+
- Git
- Virtual environment capability

### Setting Up Development Environment
1. Clone the repository and set up the environment:
```bash
# Setup the Python environment using Docker
./dev.sh start
./dev.sh test
./dev.sh test-api
./dev.sh logs
```
## Model Behavior Analysis

### Account Age and History
- **New Accounts (1-2 accounts)**
  - Low risk profile (0.1-0.6% fraud probability)
  - Model is intentionally lenient on limited history
  - Low utilization further reduces risk

### Credit Utilization Impact
- **Low Utilization (10%)**
  - Very low risk (0.1% fraud probability)
  - Indicates responsible credit management
- **High Utilization (95%)**
  - Moderate risk increase (1.3% fraud probability)
  - Not a major risk factor in isolation

### Payment History (Primary Risk Factor)
- **Accounts in Arrears**
  - No arrears: < 1% fraud probability
  - 1 account in arrears: 22.5% fraud probability
  - 3 accounts in arrears: 70.9% fraud probability (HIGH risk)
  - Most significant individual risk factor

### Account Portfolio Analysis
- **Multiple Accounts in Good Standing**
  - 15+ accounts: 6.6% base risk
  - Indicates established credit history
- **Multiple Accounts with Mixed History**
  - Many accounts with some arrears: 1.1% risk
  - Model values long-term history over isolated incidents

### Edge Case Handling
- **High-Value Accounts**
  - Very high limits ($100k+): 16.5% risk
  - Model applies additional scrutiny to large exposures
- **Small Accounts**
  - Low limits ($1k): 0.8% risk
  - Model maintains proportional risk assessment

### Key Model Characteristics
1. Payment history (arrears) is the strongest predictor
2. Account age and history provide important context
3. Credit utilization has moderate impact
4. Model balances multiple risk factors effectively
5. Edge cases are handled with reasonable risk scaling

## Cloud Architecture (AWS)
### Training Pipeline Infrastructure (SageMaker)
- **Development**: 
  - SageMaker Studio for notebook development
  - SageMaker Experiments for experiment tracking
- **Training**:
  - SageMaker Training Jobs with XGBoost built-in algorithm
  - Hyperparameter optimization with SageMaker HPO
  - Distributed training support
  - Spot instance support for cost optimization
- **Model Management**:
  - SageMaker Model Registry
  - Model versioning and lineage tracking
  - A/B testing configuration
- **Pipeline Orchestration**:
  - SageMaker Pipelines for end-to-end ML workflow
  - Automated retraining triggers
  - Model evaluation and validation steps

### Inference Pipeline Infrastructure
- **Deployment Options**:
  - SageMaker Endpoints for real-time inference
  - Auto-scaling configuration
  - Multi-model endpoints support
- **API Gateway**:
  - REST API endpoints
  - Request throttling
  - API key management
- **Monitoring**:
  - SageMaker Model Monitor
    - Data quality monitoring
    - Model quality monitoring
    - Bias drift monitoring
    - Feature attribution drift
  - CloudWatch for logs and metrics
- **Security**:
  - WAF for API protection
  - Secrets Manager for credentials
  - IAM roles for service access

### Integration Points
- **Model Registry**: SageMaker Model Registry
  - Model artifacts in S3
  - Model metadata and versioning
  - Approval workflows
- **Feature Store**: SageMaker Feature Store
  - Online and offline storage
  - Feature versioning
  - Feature sharing
- **Monitoring**: 
  - SageMaker Model Monitor
  - CloudWatch
  - Custom metrics dashboard
- **CI/CD**: 
  - AWS CodePipeline
  - SageMaker Projects
  - MLOps templates

## Key Assumptions

### Business Requirements
1. Monthly retraining schedule with capability for on-demand retraining
2. Feature engineering pipeline needed for data preparation
3. Features will be stored in SageMaker Feature Store
4. Support for A/B testing and champion-challenger
5. Inference latency requirement: < 100ms
6. Throughput requirement: 1000 TPS

### Technical Assumptions
1. Using SageMaker XGBoost container version 1.7-1
2. JSON format for API interactions
3. Both real-time and batch prediction support
4. 30-day data retention for monitoring
5. SOC 2 compliance requirements
6. Multi-language support for logs/documentation

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

### 1. ETL Pipeline
- **Data Ingestion**
  - Snowflake to S3 via AWS Glue
  - Data validation checks
  - Schema evolution handling
  
- **Feature Engineering**
  - Glue DataBrew for transformations
  - Feature Store ingestion
  - Data quality monitoring
  
- **Feature Store Management**
  - Online/Offline store sync
  - Feature groups organization
  - Access patterns optimization

### 2. Training Pipeline
- **SageMaker Pipeline Stages**
  - Feature Store data extraction
  - Data preprocessing and validation
  - Training job configuration
  - Model evaluation
  - Model registration
  - Deployment approval

- **XGBoost Configuration**
  - SageMaker built-in algorithm
  - Hyperparameters:
    - max_depth: 3
    - learning_rate: 0.3
    - objective: binary:logistic
    - eval_metric: AUC
  - Early stopping after 10 rounds
  - Distributed training enabled
  - Spot instances for cost optimization

### 3. Inference Pipeline
- **FastAPI Application**
  - RESTful endpoints for predictions
  - API key authentication
  - Health checks and model info endpoints
  - Real-time metrics tracking
  - Simple model loading with fallback to default model (xgboost.json in root directory)
  - Note: Currently using single model approach for simplicity. Model versioning can be implemented later if needed.

### 4. Model Registry
- **Version Control**
  - Timestamp-based versioning (YYYYMMDD_HHMMSS)
  - Model file storage
  - Metadata tracking:
    - Version ID
    - Training metrics
    - Feature names
    - Creation timestamp

### 5. Monitoring
- **Real-time Metrics**
  - Request count
  - Error rate
  - Prediction latency
  - Prediction value distribution
  - Automatic metric reset capability

### 6. CI/CD Pipeline
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
