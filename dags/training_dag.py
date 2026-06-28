"""
Airflow DAG: nightly churn model retraining pipeline.

Tasks
-----
generate_data  → runs scripts/generate_data.py to refresh the dataset
train_model    → runs scripts/train.py to retrain and log to MLflow

The BashOperators run inside the Airflow worker container where the
project root is mounted at /opt/airflow/project.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/airflow/project"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="churn_model_training",
    description="Nightly churn model retraining + MLflow logging",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "churn", "training"],
) as dag:

    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && python scripts/generate_data.py"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "MLFLOW_TRACKING_URI=http://mlflow:5000 "
            "DATA_PATH=data/churn.csv "
            "MODEL_OUTPUT_PATH=app/model/model.pkl "
            "python scripts/train.py"
        ),
    )

    generate_data >> train_model
