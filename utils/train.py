"""
Model Training Utility: 
This code file shows: 
1. Training a Random Forest classifier on the gold feature store.
2. Evaluating multiple models and saving the best one to the model bank.

Model bank structure:
    model_bank/
    └── YYYY_MM_DD/
        ├── model.pkl          # consists of the trained model object
        └── metadata.json      # training metadata (accuracy, AUC, features)

"""

import os
import glob
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)


def load_gold_tables(gold_dir, start_date, end_date):
    """
    Load the feature store and label store parquet files within the training date range and merge them.
    
    Shall only load data up to end_date to prevent temporal leakage.
    """
    feature_files = sorted(glob.glob(os.path.join(gold_dir, "feature_store", "*.parquet")))
    label_files   = sorted(glob.glob(os.path.join(gold_dir, "label_store",   "*.parquet")))

    # Filter by date range
    def in_range(f):
        name = os.path.basename(f).replace(".parquet", "").replace("_", "-")
        return start_date <= name <= end_date

    feature_files = [f for f in feature_files if in_range(f)]
    label_files   = [f for f in label_files   if in_range(f)]

    if not feature_files or not label_files:
        raise ValueError(f"No gold data found between {start_date} and {end_date}")

    df_features = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)
    df_labels   = pd.concat([pd.read_parquet(f) for f in label_files],   ignore_index=True)

    # Merge on customer_id
    df = df_features.merge(df_labels[["customer_id", "label"]], on="customer_id", how="inner")
    return df


def prepare_features(df):
    """
    Drop all the non-feature columns and then prepare X, y for sklearn.
    Fill remaining nulls with 0.
    """
    drop_cols = ["customer_id", "loan_id", "snapshot_date",
                 "loan_start_date", "label_def", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["label"]
    return X, y, list(X.columns)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Training multiple models and returning the best one by AUC.
    
    Models which are going to be compared here are:
    - Logistic Regression (baseline)
    - Random Forest
    - Gradient Boosting
    """
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model":     model,
            "accuracy":  round(accuracy_score(y_test, y_pred),   4),
            "auc":       round(roc_auc_score(y_test, y_proba),   4),
            "precision": round(precision_score(y_test, y_pred),  4),
            "recall":    round(recall_score(y_test, y_pred),     4),
            "f1":        round(f1_score(y_test, y_pred),         4),
        }
        print(f"  {name}: AUC={results[name]['auc']} | Acc={results[name]['accuracy']}")

    # Select the best model defined by AUC scoring
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  Best model: {best_name} (AUC={results[best_name]['auc']})")
    return best_name, results[best_name]


def save_model(model, model_name, metrics, feature_cols, model_bank_dir, training_date,
               top_importances=None):
    """
    Save model and metadata to the model bank.

    model_bank/
    └── 2024_01_01/
        ├── model.pkl
        └── metadata.json
    """
    model_dir = os.path.join(model_bank_dir, training_date.replace("-", "_"))
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "training_date":    training_date,
        "model_name":       model_name,
        "n_features":       len(feature_cols),
        "feature_cols":     feature_cols,
        "top_importances":  top_importances or {},
        "metrics":          {k: v for k, v in metrics.items() if k != "model"},
        "saved_at":         datetime.utcnow().isoformat(),
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model saved to: {model_dir}")
    return model_dir


def run_training(gold_dir, model_bank_dir, training_date, start_date="2023-01-01"):
    """
    Main training function shall be called by the Airflow DAG.

    Steps:
    1. Loading the gold tables from start_date up to training_date
    2. Preparing features
    3. Feature selection (low-variance → high-correlation → RF importance)
    4. Training and evaluating multiple models on selected features
    5. Saving the best model and metadata to the model bank
    """
    from utils.feature_selection import select_features

    print(f"[Train] Loading gold data from {start_date} to {training_date}")
    df = load_gold_tables(gold_dir, start_date, training_date)
    print(f"[Train] Loaded {len(df)} rows, {df['label'].mean():.1%} default rate")

    X, y, _ = prepare_features(df)
    print(f"[Train] Raw feature matrix: {X.shape}")

    # Temporal split: train on earlier data, test on most recent 20%
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Feature selection — fit on train only to prevent leakage
    X_train_sel, selected_cols, importances = select_features(X_train, y_train, top_n=40)
    X_test_sel = X_test[selected_cols]
    print(f"[Train] Selected {len(selected_cols)} features after selection")

    print("[Train] Evaluating models...")
    best_name, best_metrics = train_and_evaluate(X_train_sel, X_test_sel, y_train, y_test)

    # Save top 10 importances alongside model for reporting
    top_importances = importances.head(10).to_dict()

    model_dir = save_model(
        model          = best_metrics["model"],
        model_name     = best_name,
        metrics        = best_metrics,
        feature_cols   = selected_cols,
        model_bank_dir = model_bank_dir,
        training_date  = training_date,
        top_importances= top_importances,
    )

    return model_dir


def get_latest_model(model_bank_dir):
    model_dirs = sorted(glob.glob(os.path.join(model_bank_dir, "*")))
    if not model_dirs:
        raise FileNotFoundError("No models found in model bank!")

    latest_dir = model_dirs[-1]
    model_path = os.path.join(latest_dir, "model.pkl")
    meta_path  = os.path.join(latest_dir, "metadata.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print(f"[Model] Loaded: {metadata['model_name']} trained on {metadata['training_date']}")
    return model, metadata
