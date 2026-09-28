# End-to-End Loan Default ML Pipeline with Airflow, Drift Monitoring and Retraining Rules

SMU Machine Learning Engineering course, Assignment 2 (2026). Extends the [Assignment 1 data pipeline](https://github.com/4h4n4-01/loan-default-data-pipeline) into a monthly, orchestrated train → predict → monitor loop.

## Design decisions

- **Industry-standard label:** default = 30+ days past due at month six on book (dpd ≥ 30 at mob = 6).
- **No leakage:** clickstream features restricted to dates before each loan starts; one parquet partition per month, so every run sees only data up to its month.
- **Privacy:** name and SSN removed in the silver layer and never reach the gold feature store.
- **Governance fixed in advance:** retrain every six months (January and July), or earlier if PSI ≥ 0.25 or AUC < 0.65.

## Pipeline

A 24-month backfill (Jan 2023 – Dec 2024); each monthly run executes eight Airflow tasks:

```
bronze_ingestion → silver_cleaning → gold_label_store → gold_feature_store
→ model_training → model_prediction → model_monitoring → visualise_monitoring
```

Three models are compared by AUC at each retraining point; the best is stored in `model_bank/` with metadata.

## Results

| | |
|---|---|
| Best model | Gradient boosting, trained July 2024 |
| AUC | 0.901 |
| Drift (PSI) | Stable across most months; a spike at the retraining point, as expected |
| Reliability | 24/24 DAG runs succeeded; 0 failures across 192 task executions |

The monitoring dashboard is written to `reports/monitoring_dashboard.png`.

## Run

1. Start Docker Desktop.
2. `docker-compose build`
3. `docker-compose up`
4. Open Airflow at `http://localhost:8080` (local default credentials are set in `docker-compose.yaml`).
5. The DAG backfills Jan 2023 – Dec 2024 automatically.

## Repository

```
dags/ml_pipeline_dag.py      Airflow DAG (8 tasks)
utils/data_pipeline.py       bronze / silver / gold processing
utils/train.py               training and model selection by AUC
utils/predict.py             inference with the stored best model
utils/monitor.py             PSI + AUC monitoring and dashboard
data/                        raw CSV sources (4 files)
Dockerfile, docker-compose.yaml, requirements.txt
```
