"""
Prediction Utility: 
This code file helps to load the best model from the model bank and makes predictions
for a given snapshot date. The predictions are then saved as a gold table.

"""

from importlib.metadata import metadata
import os
import glob
import json
import pandas as pd
import numpy as np


def load_features_for_date(gold_dir, snapshot_date):
    """Load feature store partition for a specific date."""
    partition_name = snapshot_date.replace("-", "_") + ".parquet"
    filepath = os.path.join(gold_dir, "feature_store", partition_name)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No feature store found for {snapshot_date}")

    return pd.read_parquet(filepath)


def load_labels_for_date(gold_dir, snapshot_date):
    """Load label store partition for a specific date."""
    partition_name = snapshot_date.replace("-", "_") + ".parquet"
    filepath = os.path.join(gold_dir, "label_store", partition_name)

    if not os.path.exists(filepath):
        return None

    return pd.read_parquet(filepath)


def run_prediction(model, metadata, gold_dir, snapshot_date):
    """
    Run inference for a given snapshot date.
    
    Steps:
    1. Loading feature store for the date
    2. Aligning features to what the model was trained on
    3. Generating predictions and probabilities
    4. Saving predictions as gold table partition
    
    Returns a DataFrame with:
    - customer_id
    - prediction (0 or 1)
    - probability (0.0 to 1.0)
    - model_name, training_date (for traceability)
    - snapshot_date
    """
    print(f"[Predict] Running inference for {snapshot_date}")

    df_features = load_features_for_date(gold_dir, snapshot_date)

    if df_features.empty:
        print(f"[Predict] No features found for {snapshot_date}, skipping")
        return None

    # Align features to training columns
    feature_cols = metadata["feature_cols"]
    X = df_features.reindex(columns=feature_cols).fillna(0)

    # Generate predictions
    predictions  = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Build output DataFrame
    df_pred = pd.DataFrame({
        "customer_id":   df_features["customer_id"].values,
        "snapshot_date": snapshot_date,
        "prediction":    predictions,
        "probability":   probabilities.round(4),
        "model_name":    metadata["model_name"],
        "training_date": metadata["training_date"],
    })
    # Only include loan_id when the feature store actually has it; setting it to
    # None/NaN causes the monitor join to match zero rows and silently drops AUC.
    if "loan_id" in df_features.columns:
        df_pred.insert(1, "loan_id", df_features["loan_id"].values)

    # Save to gold table
    partition_name = snapshot_date.replace("-", "_") + ".parquet"
    out_path = os.path.join(gold_dir, "predictions", partition_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_pred.to_parquet(out_path, index=False)

    print(f"[Predict] Saved {len(df_pred)} predictions -> {out_path}")
    print(f"[Predict] Default rate predicted: {predictions.mean():.1%}")

    return df_pred
