"""
CS611 Assignment 2: ML Pipeline DAG
This Airflow DAG orchestrates the full end-to-end ML pipeline.

DAG Schedule: Monthly (runs on the 1st of each month)
Backfill: Supports backfilling from 2023-01-01 to 2024-12-01

Task Flow:
    bronze --> silver --> gold_labels --> gold_features --> train --> predict --> monitor --> visualise

Each task runs for a specific snapshot_date which is the execution date.
This means when backfilling, each month is processed independently.

"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os
sys.path.insert(0, "/opt/airflow")

# DAG Configuration 

DEFAULT_ARGS = {
    "owner":            "ahana",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

DAG_CONFIG = {
    "dag_id":      "CS611_ML_Pipeline",
    "description": "End-to-end ML pipeline for loan default prediction",
    "schedule":    "0 0 1 * *",        # Run on 1st of every month
    "start_date":  datetime(2023, 1, 1),
    "end_date":    datetime(2024, 12, 1),
    "catchup":     True,               # Enable backfilling
    "max_active_runs": 1,
    "tags":["data-ingestion", "data-cleaning", "feature-engineering", "model-training", "inference", "model-monitoring"],
}

# Paths 
BASE_DIR       = "/opt/airflow"
DATA_DIR       = os.path.join(BASE_DIR, "data")
DATAMART_DIR   = os.path.join(BASE_DIR, "datamart")
MODEL_BANK_DIR = os.path.join(BASE_DIR, "model_bank")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")

BRONZE_DIR  = os.path.join(DATAMART_DIR, "bronze")
SILVER_DIR  = os.path.join(DATAMART_DIR, "silver")
GOLD_DIR    = os.path.join(DATAMART_DIR, "gold")

# Training starts after 6 months of data (mob=6 label definition needs 6 months)
TRAINING_START_DATE = "2023-01-01"
TRAINING_CUTOFF     = "2023-06-01"   # The first date that we have enough data to train


# Task Functions

def task_bronze(snapshot_date, **context):
    import pyspark
    from utils.data_pipeline import process_bronze_table

    spark = (
        pyspark.sql.SparkSession.builder
        .appName(f"bronze_{snapshot_date}")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"[Bronze] Processing snapshot: {snapshot_date}")

    sources = {
        "clickstream": "feature_clickstream.csv",
        "attributes":  "features_attributes.csv",
        "financials":  "features_financials.csv",
        "lms":         "lms_loan_daily.csv",
    }

    for table_name, filename in sources.items():
        filepath = os.path.join(DATA_DIR, filename)
        process_bronze_table(table_name, filepath, BRONZE_DIR, snapshot_date, spark)
        print(f"[Bronze] {table_name}")

    spark.stop()


def task_silver(snapshot_date, **context):
    import pyspark
    from utils.data_pipeline import process_silver_table

    spark = (
        pyspark.sql.SparkSession.builder
        .appName(f"silver_{snapshot_date}")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"[Silver] Processing snapshot: {snapshot_date}")

    for table_name in ["clickstream", "attributes", "financials", "lms"]:
        process_silver_table(table_name, BRONZE_DIR, SILVER_DIR, snapshot_date, spark)
        print(f"[Silver] {table_name}")

    spark.stop()


def task_gold_labels(snapshot_date, **context):
    if snapshot_date < TRAINING_CUTOFF:
        print(f"[Gold Labels] Skipping {snapshot_date} - need mob=6 data, cutoff is {TRAINING_CUTOFF}")
        return

    import pyspark
    from utils.data_pipeline import build_label_store

    spark = (
        pyspark.sql.SparkSession.builder
        .appName(f"gold_labels_{snapshot_date}")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"[Gold Labels] Building label store for {snapshot_date}")
    df = build_label_store(SILVER_DIR, GOLD_DIR, snapshot_date, spark)
    print(f"[Gold Labels] {df.count()} labels saved")
    spark.stop()


def task_gold_features(snapshot_date, **context):
    import pyspark
    from utils.data_pipeline import build_feature_store

    spark = (
        pyspark.sql.SparkSession.builder
        .appName(f"gold_features_{snapshot_date}")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"[Gold Features] Building feature store for {snapshot_date}")
    df = build_feature_store(SILVER_DIR, GOLD_DIR, snapshot_date, spark)
    print(f"[Gold Features] {df.count()} feature rows saved")
    spark.stop()


def task_eda(snapshot_date, **context):
    if snapshot_date < TRAINING_CUTOFF:
        print(f"[EDA] Skipping {snapshot_date} - not enough data yet")
        return

    month = int(snapshot_date.split("-")[1])
    if month not in [1, 7]:
        print(f"[EDA] Skipping {snapshot_date} - EDA runs only at retraining months (Jan/Jul)")
        return

    from utils.eda import run_eda

    print(f"[EDA] Running exploratory data analysis for {snapshot_date}")
    stats = run_eda(GOLD_DIR, REPORTS_DIR, training_date=snapshot_date)
    if stats:
        print(f"[EDA] Records={stats['n_total']} | Default rate={stats['default_rate']:.2%} "
              f"| Features={stats['n_features']} | Top feature={stats['top_feature']}")


def task_train(snapshot_date, **context):
    if snapshot_date < TRAINING_CUTOFF:
        print(f"[Train] Skipping {snapshot_date} - not enough data yet")
        return

    # Only retrain every 6 months
    month = int(snapshot_date.split("-")[1])
    if month not in [1, 7]:
        print(f"[Train] Skipping {snapshot_date} - not a retraining month (Jan/Jul)")
        return

    from utils.train import run_training

    print(f"[Train] Training model for snapshot: {snapshot_date}")
    model_dir = run_training(
        gold_dir       = GOLD_DIR,
        model_bank_dir = MODEL_BANK_DIR,
        training_date  = snapshot_date,
        start_date     = TRAINING_START_DATE,
    )
    print(f"[Train] Model saved to {model_dir}")


def task_predict(snapshot_date, **context):
    if snapshot_date < TRAINING_CUTOFF:
        print(f"[Predict] Skipping {snapshot_date} - model not yet trained")
        return

    # Skip if no model has been trained yet (first training is at 2023-07-01)
    import glob as _glob
    if not _glob.glob(os.path.join(MODEL_BANK_DIR, "*", "model.pkl")):
        print(f"[Predict] Skipping {snapshot_date} - no model in model bank yet")
        return

    from utils.train import get_latest_model
    from utils.predict import run_prediction

    model, metadata = get_latest_model(MODEL_BANK_DIR)
    df_pred = run_prediction(model, metadata, GOLD_DIR, snapshot_date)

    if df_pred is not None:
        print(f"[Predict] {len(df_pred)} predictions saved")


def task_monitor(snapshot_date, **context):
    if snapshot_date < TRAINING_CUTOFF:
        print(f"[Monitor] Skipping {snapshot_date} - predictions not yet available")
        return

    # Skip if no predictions exist yet
    import glob as _glob
    if not _glob.glob(os.path.join(GOLD_DIR, "predictions", "*.parquet")):
        print(f"[Monitor] Skipping {snapshot_date} - no predictions available yet")
        return

    from utils.monitor import run_monitoring

    result = run_monitoring(GOLD_DIR, MODEL_BANK_DIR, snapshot_date)
    if result:
        print(f"[Monitor] PSI={result.get('psi')} | AUC={result.get('auc')}")


def task_visualise(**context):
    import glob as _glob
    import pandas as pd
    from utils.monitor import run_monitoring, generate_monitoring_dashboard

    # Labels lag 6 months behind predictions, so AUC is always None at inline-run time.
    # Re-check every past monitoring month here: if AUC is still missing but labels
    # have since been generated (by later pipeline months), backfill it now.
    mon_files = sorted(_glob.glob(os.path.join(GOLD_DIR, "monitoring", "*.parquet")))
    for mf in mon_files:
        df = pd.read_parquet(mf)
        if "auc" not in df.columns or df["auc"].isna().all():
            date = df["snapshot_date"].iloc[0]
            print(f"[Visualise] Backfilling AUC for {date}")
            run_monitoring(GOLD_DIR, MODEL_BANK_DIR, date)

    output_path = generate_monitoring_dashboard(GOLD_DIR, REPORTS_DIR)
    if output_path:
        print(f"[Visualise] Dashboard saved to {output_path}")


# DAG Definition

with DAG(
    default_args = DEFAULT_ARGS,
    **DAG_CONFIG,
) as dag:

    # Get snapshot date from Airflow execution date
    snapshot_date = "{{ ds }}"   # Airflow injects the execution date as YYYY-MM-DD

    # Define tasks
    t_bronze = PythonOperator(
        task_id         = "bronze_ingestion",
        python_callable = task_bronze,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_silver = PythonOperator(
        task_id         = "silver_cleaning",
        python_callable = task_silver,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_gold_labels = PythonOperator(
        task_id         = "gold_label_store",
        python_callable = task_gold_labels,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_gold_features = PythonOperator(
        task_id         = "gold_feature_store",
        python_callable = task_gold_features,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_eda = PythonOperator(
        task_id         = "eda_analysis",
        python_callable = task_eda,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_train = PythonOperator(
        task_id         = "model_training",
        python_callable = task_train,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_predict = PythonOperator(
        task_id         = "model_prediction",
        python_callable = task_predict,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_monitor = PythonOperator(
        task_id         = "model_monitoring",
        python_callable = task_monitor,
        op_kwargs       = {"snapshot_date": snapshot_date},
    )

    t_visualise = PythonOperator(
        task_id         = "visualise_monitoring",
        python_callable = task_visualise,
    )

    # Task Dependencies
    # bronze --> silver --> [gold_labels, gold_features] --> eda_analysis --> train --> predict --> monitor --> visualise
    t_bronze >> t_silver >> [t_gold_labels, t_gold_features]
    [t_gold_labels, t_gold_features] >> t_eda >> t_train >> t_predict >> t_monitor >> t_visualise
