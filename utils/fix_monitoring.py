"""
One-shot script to recompute all monitoring parquet files with correct AUC values
and regenerate the dashboard.

Run from inside the Airflow Docker container:
    python /opt/airflow/utils/fix_monitoring.py

This script re-runs calculate_performance_metrics with the fixed join logic
(customer_id only when loan_id is NaN) and overwrites each monitoring partition.
"""

import os
import sys
import glob
import json

import pandas as pd
import numpy as np

sys.path.insert(0, "/opt/airflow")

GOLD_DIR       = "/opt/airflow/datamart/gold"
MODEL_BANK_DIR = "/opt/airflow/model_bank"
REPORTS_DIR    = "/opt/airflow/reports"


def recompute_monitoring():
    monitor_files = sorted(glob.glob(os.path.join(GOLD_DIR, "monitoring", "*.parquet")))
    if not monitor_files:
        print("[Fix] No monitoring files found — nothing to do")
        return

    # Load all label partitions once
    all_label_files = glob.glob(os.path.join(GOLD_DIR, "label_store", "*.parquet"))
    if not all_label_files:
        print("[Fix] No label store files found — cannot compute AUC")
        return

    df_labels_all = pd.concat(
        [pd.read_parquet(f) for f in all_label_files], ignore_index=True
    )
    df_labels_all = df_labels_all.drop_duplicates(subset=["customer_id"])
    print(f"[Fix] Loaded {len(df_labels_all)} unique labelled customers")

    # Load latest model metadata (for model_name / training_date fields)
    meta_path = sorted(glob.glob(os.path.join(MODEL_BANK_DIR, "*", "metadata.json")))[-1]
    with open(meta_path) as f:
        metadata = json.load(f)

    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        precision_score, recall_score, f1_score,
    )

    for mf in monitor_files:
        df_mon = pd.read_parquet(mf)
        snapshot_date = df_mon["snapshot_date"].iloc[0]

        pred_path = os.path.join(
            GOLD_DIR, "predictions",
            snapshot_date.replace("-", "_") + ".parquet"
        )
        if not os.path.exists(pred_path):
            print(f"[Fix] No predictions for {snapshot_date}, skipping")
            continue

        df_pred = pd.read_parquet(pred_path)

        # --- fixed join: customer_id only when loan_id is absent / all-NaN ---
        join_keys = ["customer_id"]
        if ("loan_id" in df_pred.columns and "loan_id" in df_labels_all.columns
                and df_pred["loan_id"].notna().any()):
            join_keys = ["customer_id", "loan_id"]

        df = df_pred.merge(df_labels_all[join_keys + ["label"]], on=join_keys, how="inner")

        if df.empty or df["label"].nunique() < 2:
            print(f"[Fix] {snapshot_date}: no matched labels yet, skipping AUC")
            continue

        y_true = df["label"]
        y_pred = df["prediction"]
        y_prob = df["probability"]

        perf = {
            "accuracy":               round(accuracy_score(y_true, y_pred),  4),
            "auc":                    round(roc_auc_score(y_true, y_prob),    4),
            "precision":              round(precision_score(y_true, y_pred),  4),
            "recall":                 round(recall_score(y_true, y_pred),     4),
            "f1":                     round(f1_score(y_true, y_pred),         4),
            "default_rate_actual":    round(float(y_true.mean()),             4),
            "default_rate_predicted": round(float(y_pred.mean()),             4),
        }

        for k, v in perf.items():
            df_mon[k] = v

        df_mon.to_parquet(mf, index=False)
        print(f"[Fix] {snapshot_date}: AUC={perf['auc']} | matched {len(df)} customers")

    print("\n[Fix] All monitoring files updated — regenerating dashboard...")
    from utils.monitor import generate_monitoring_dashboard
    out = generate_monitoring_dashboard(GOLD_DIR, REPORTS_DIR)
    print(f"[Fix] Dashboard saved to {out}")


if __name__ == "__main__":
    recompute_monitoring()
