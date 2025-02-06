import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ModelStep, RegisterModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.xgboost import XGBoost
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model import Model
from datetime import datetime
import os
import json

def create_pipeline(
    role_arn: str,
    bucket_name: str,
    pipeline_name: str = "fraud-detection-pipeline",
    region: str = "us-east-1",
    model_package_group_name: str = "fraud-detection-models",
    schedule_expression: str = "rate(30 days)"
):
    """
    Create an end-to-end SageMaker pipeline for fraud detection model training and deployment.
    
    Args:
        role_arn (str): ARN of the IAM role with SageMaker permissions
        bucket_name (str): S3 bucket name for storing artifacts
        pipeline_name (str): Name of the pipeline
        region (str): AWS region
        model_package_group_name (str): Name for the model package group in Model Registry
        schedule_expression (str): Schedule expression for pipeline execution
    """
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Pipeline parameters
    timestamp = ParameterString(
        name="ExecutionTime",
        default_value=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge"
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    
    min_auc_threshold = ParameterFloat(
        name="MinAUCThreshold",
        default_value=0.8
    )
    
    # S3 paths
    base_s3_path = f"s3://{bucket_name}/fraud-detection/{timestamp}"
    
    # Create preprocessing script
    preprocessing_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

def preprocess_data():
    input_data_path = "/opt/ml/processing/input"
    train_path = "/opt/ml/processing/train"
    validation_path = "/opt/ml/processing/validation"
    test_path = "/opt/ml/processing/test"
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Read and preprocess data
    df = pd.read_csv(f"{input_data_path}/fraud_data.csv")
    
    # Feature engineering
    df['credit_utilization'] = df['total_balance'] / df['total_credit_limit']
    df['credit_utilization'] = df['credit_utilization'].fillna(0)
    
    # Feature list
    features = [
        'number_of_open_accounts',
        'total_credit_limit',
        'total_balance',
        'number_of_accounts_in_arrears',
        'credit_utilization'
    ]
    
    # Split data into features and target
    X = df[features]
    y = df['is_fraud']
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Save datasets
    train_data = pd.concat([y_train, X_train], axis=1)
    val_data = pd.concat([y_val, X_val], axis=1)
    test_data = pd.concat([y_test, X_test], axis=1)
    
    train_data.to_csv(f"{train_path}/train.csv", index=False)
    val_data.to_csv(f"{validation_path}/validation.csv", index=False)
    test_data.to_csv(f"{test_path}/test.csv", index=False)
    
    # Save feature metadata
    feature_metadata = {
        'features': features,
        'target': 'is_fraud',
        'train_shape': train_data.shape,
        'validation_shape': val_data.shape,
        'test_shape': test_data.shape
    }
    
    with open("/opt/ml/processing/metadata/feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f)

if __name__ == "__main__":
    preprocess_data()
"""
    
    # Create training script
    training_code = """
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np

def train_and_evaluate():
    # Load data
    train_data = pd.read_csv("/opt/ml/input/data/train/train.csv")
    val_data = pd.read_csv("/opt/ml/input/data/validation/validation.csv")
    
    # Load feature metadata
    with open("/opt/ml/input/data/metadata/feature_metadata.json", "r") as f:
        feature_metadata = json.load(f)
    
    features = feature_metadata['features']
    target = feature_metadata['target']
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data[target]
    X_val = val_data[features]
    y_val = val_data[target]
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    
    # Training parameters
    params = {
        'max_depth': 3,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'aucpr'],
        'tree_method': 'hist',  # For faster training
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle class imbalance
    }
    
    # Train model
    num_round = 100
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    bst = xgb.train(
        params,
        dtrain,
        num_round,
        evallist,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Evaluate model
    val_predictions = bst.predict(dval)
    auc_score = roc_auc_score(y_val, val_predictions)
    avg_precision = average_precision_score(y_val, val_predictions)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, val_predictions)
    
    # Feature importance
    feature_importance = bst.get_score(importance_type='gain')
    
    # Save metrics
    metrics = {
        'validation_auc': float(auc_score),
        'validation_avg_precision': float(avg_precision),
        'validation_size': len(y_val),
        'training_size': len(y_train),
        'feature_count': len(features),
        'best_iteration': bst.best_iteration,
        'feature_importance': feature_importance
    }
    
    # Save model and artifacts
    model_path = "/opt/ml/model"
    os.makedirs(model_path, exist_ok=True)
    
    bst.save_model(f"{model_path}/xgboost.json")
    
    with open(f"{model_path}/metrics.json", "w") as f:
        json.dump(metrics, f)
    
    # Save feature metadata with model
    with open(f"{model_path}/feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f)

if __name__ == "__main__":
    train_and_evaluate()
"""
    
    # Save scripts
    os.makedirs("code", exist_ok=True)
    with open("code/preprocessing.py", "w") as f:
        f.write(preprocessing_code)
    with open("code/train.py", "w") as f:
        f.write(training_code)
    
    # Processing step
    processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.0-1"
        ),
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        command=["python3"],
        base_job_name=f"{pipeline_name}-preprocessing",
        sagemaker_session=sagemaker_session
    )
    
    processing_step = ProcessingStep(
        name="PreprocessFraudData",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"{base_s3_path}/input",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"{base_s3_path}/train"
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=f"{base_s3_path}/validation"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"{base_s3_path}/test"
            ),
            ProcessingOutput(
                output_name="metadata",
                source="/opt/ml/processing/metadata",
                destination=f"{base_s3_path}/metadata"
            )
        ],
        code="code/preprocessing.py"
    )
    
    # Training step
    xgb_estimator = XGBoost(
        entry_point="train.py",
        source_dir="code",
        framework_version="1.5-1",
        hyperparameters={
            "max_depth": 3,
            "eta": 0.3,
            "objective": "binary:logistic",
            "num_round": 100
        },
        role=role_arn,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"{base_s3_path}/model",
        sagemaker_session=sagemaker_session
    )
    
    training_step = TrainingStep(
        name="TrainFraudModel",
        estimator=xgb_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "metadata": sagemaker.inputs.TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["metadata"].S3Output.S3Uri,
                content_type="application/json"
            )
        }
    )
    
    # Evaluation step to check model quality
    eval_metrics = PropertyFile(
        name="EvaluationReport",
        output_name="metrics",
        path="metrics.json"
    )
    
    # Model metrics for model registry
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{base_s3_path}/evaluation/metrics.json",
            content_type="application/json"
        )
    )
    
    # Register model step
    model = Model(
        image_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=sagemaker_session
    )
    
    register_step = RegisterModel(
        name="RegisterFraudModel",
        estimator=xgb_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )
    
    # Condition step to check model quality
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=eval_metrics.get_output()["validation_auc"],
        right=min_auc_threshold
    )
    
    condition_step = ConditionStep(
        name="CheckModelQuality",
        conditions=[cond_gte],
        if_steps=[register_step],
        else_steps=[]
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            timestamp,
            training_instance_type,
            model_approval_status,
            min_auc_threshold
        ],
        steps=[processing_step, training_step, condition_step],
        sagemaker_session=sagemaker_session
    )
    
    # Create pipeline if it doesn't exist
    pipeline.upsert(role_arn=role_arn)
    
    # Schedule pipeline execution
    events_client = boto3.client('events', region_name=region)
    
    rule_name = f"{pipeline_name}-schedule"
    
    events_client.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,
        State='ENABLED',
        Description=f'Schedule for SageMaker Pipeline {pipeline_name}'
    )
    
    target = {
        'Id': f"{pipeline_name}-target",
        'Arn': f"arn:aws:sagemaker:{region}:{sagemaker_session.account_id}:pipeline/{pipeline_name}",
        'RoleArn': role_arn,
        'Input': json.dumps({
            'PipelineParameters': [
                {
                    'Name': 'ExecutionTime',
                    'Value': '$.time'
                }
            ]
        })
    }
    
    events_client.put_targets(
        Rule=rule_name,
        Targets=[target]
    )
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    ROLE_ARN = "arn:aws:iam::ACCOUNT_ID:role/service-role/sagemaker-role"  # Replace with your role ARN
    BUCKET_NAME = "your-bucket-name"  # Replace with your bucket name
    REGION = "us-east-1"
    
    pipeline = create_pipeline(
        role_arn=ROLE_ARN,
        bucket_name=BUCKET_NAME,
        region=REGION
    ) 