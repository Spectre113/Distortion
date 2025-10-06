from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"
sys.path.append(str(NOTEBOOKS_PATH))

from data_engineering import run_inference_pipeline

default_args = {
    "owner": "airflow",
    "retries": 0,
}

with DAG(
    dag_id="waveunet_inference_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 10, 6),
    schedule="*/20 * * * *",
    catchup=False,
    tags=["mlops", "inference", "audio"],
) as dag:

    run_inference = PythonOperator(
        task_id="run_waveunet_inference",
        python_callable=run_inference_pipeline,
        op_kwargs={
            "input_root": "/mnt/data/to_process",
            "model_checkpoint": "/mnt/data/checkpoints/best_model.pt",
            "output_root": "/mnt/data/inference_out",
            "chunk_duration": 3.0,
            "sample_rate": 22050,
            "batch_size": 8,
            "device_str": "cuda",
            "glob_patterns": ["**/*.wav"],
            "max_files": 200,
        },
    )

    run_inference