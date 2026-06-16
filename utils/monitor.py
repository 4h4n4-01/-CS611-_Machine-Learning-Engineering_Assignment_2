"""
Model Monitoring Utility:
This code file monitors model performance and stability over time.

Two types of monitoring that is done here:
1. Performancemonitoring which ensure if the model is still accurate?
   - Requires actual labels i.e. known outcomes
   - Metrics: AUC, accuracy, precision, recall, F1

2. Stability monitoring which ensure if the input data distribution is changing?
   - Uses PSI (Population Stability Index)
   - Compares current score distribution to training distribution
   - Does not require labels and can be done immediately after prediction

PSI interpretation:
   PSI < 0.10:  No significant change, model is stable
   PSI < 0.25:  Minor shift, monitor closely
   PSI >= 0.25: Major shift, consider retraining

Model Governance SOP:
- Retrain if: AUC drops below 0.65 OR PSI >= 0.25
- Review if: AUC between 0.65-0.70 OR PSI between 0.10-0.25
- No action if: AUC >= 0.70 AND PSI < 0.10
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend - must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)


# PSI Calculation

def calculate_psi(baseline_scores, current_scores, n_bins=10):
    """
    Calculates the Population Stability Index between baseline and current scores.

    PSI = sum((current% - baseline%) * ln(current% / baseline%))

    baseline_scores: predicted probabilities from previous month
    current_scores:  predicted probabilities from current month

    PSI < 0.10  = Stable
    PSI < 0.25  = Minor Shift - Monitor Closely
    PSI >= 0.25 = Major Shift - Retrain Model
    """
    # Create bins from baseline distribution
    bins = np.percentile(baseline_scores, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate proportions in each bin
    baseline_counts = np.histogram(baseline_scores, bins=bins)[0]
    current_counts  = np.histogram(current_scores,  bins=bins)[0]

    # Convert to proportions, add small epsilon to avoid division by zero
    eps = 1e-6
    baseline_pct = baseline_counts / (len(baseline_scores) + eps) + eps
    current_pct  = current_counts  / (len(current_scores)  + eps) + eps

    # PSI formula
    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    psi = np.sum(psi_values)
    return round(float(psi), 4)


def interpret_psi(psi):
    if psi < 0.10:
        return "Stable", "green"
    elif psi < 0.25:
        return "Minor Shift - Monitor Closely", "orange"
    else:
        return "Major Shift - Retrain Model", "red"


# Performance Monitoring

def calculate_performance_metrics(df_pred, df_labels):
    """
    Calculate performance metrics by joining predictions with actual labels.
    Joins across ALL label partitions not just the current snapshot date.
    This is needed because labels are derived at mob=6 which is a different
    date from when predictions are made at loan application time.
    """
    
    join_keys = ["customer_id"]

    # Only join on loan_id if predictions actually have non-null loan_ids.
    # The feature store has no loan_id, so predict.py fills the column with None/NaN.
    # Joining on a NaN column produces zero matches, silently killing AUC for every month.
    if ("loan_id" in df_pred.columns and "loan_id" in df_labels.columns
            and df_pred["loan_id"].notna().any()):
        join_keys = ["customer_id", "loan_id"]

    df = df_pred.merge(
        df_labels[join_keys + ["label"]],
        on=join_keys,
        how="inner"
    )

    if df.empty or df["label"].nunique() < 2:
        return None

    y_true = df["label"]
    y_pred = df["prediction"]
    y_prob = df["probability"]

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),          4),
        "auc":       round(roc_auc_score(y_true, y_prob),           4),
        "precision": round(precision_score(y_true, y_pred),         4),
        "recall":    round(recall_score(y_true, y_pred),            4),
        "f1":        round(f1_score(y_true, y_pred),                4),
        "default_rate_actual":    round(float(y_true.mean()),       4),
        "default_rate_predicted": round(float(y_pred.mean()),       4),
    }


# Main Monitoring Function

def run_monitoring(gold_dir, model_bank_dir, snapshot_date):
    """
    Run monitoring for a given snapshot date.

    Steps:
    1. Loading predictions for this date
    2. Calculating PSI vs PREVIOUS MONTH predictions (not vs training baseline)
       This gives a more meaningful month-on-month stability measure
    3. Calculating performance metrics by joining with ALL available labels
       across all partitions (because mob=6 labels are on different dates)
    4. Saving monitoring results as gold table partition
    """
    print(f"[Monitor] Running monitoring for {snapshot_date}")

    # Load predictions for this date
    pred_path = os.path.join(gold_dir, "predictions",
                             snapshot_date.replace("-", "_") + ".parquet")
    if not os.path.exists(pred_path):
        print(f"[Monitor] No predictions found for {snapshot_date}, skipping")
        return None

    df_pred = pd.read_parquet(pred_path)

    # Load model metadata
    meta_path = sorted(glob.glob(os.path.join(model_bank_dir, "*", "metadata.json")))[-1]
    with open(meta_path) as f:
        metadata = json.load(f)

    # FIX 1: PSI vs PREVIOUS MONTH (not vs training baseline)
    # This gives meaningful month-on-month stability instead of
    # always comparing to the very first month which inflates PSI
    all_pred_files = sorted(glob.glob(os.path.join(gold_dir, "predictions", "*.parquet")))
    current_file = snapshot_date.replace("-", "_") + ".parquet"
    current_idx = next((i for i, f in enumerate(all_pred_files)
                        if os.path.basename(f) == current_file), None)

    psi = None
    if current_idx is not None and current_idx > 0:
        df_baseline = pd.read_parquet(all_pred_files[current_idx - 1])
        psi = calculate_psi(
            df_baseline["probability"].values,
            df_pred["probability"].values
        )
        psi_label, _ = interpret_psi(psi)
        print(f"[Monitor] PSI = {psi} -> {psi_label}")

    # FIX 2: Load ALL label partitions and join on customer_id
    # Labels are generated at mob=6 which is a different snapshot_date
    # from when predictions are made, so we join across all dates
    perf_metrics = None
    all_label_files = glob.glob(os.path.join(gold_dir, "label_store", "*.parquet"))
    if all_label_files:
        df_labels = pd.concat(
            [pd.read_parquet(f) for f in all_label_files], ignore_index=True
        )
        df_labels = df_labels.drop_duplicates(subset=["customer_id"])
        perf_metrics = calculate_performance_metrics(df_pred, df_labels)
        if perf_metrics:
            print(f"[Monitor] AUC={perf_metrics['auc']} | Acc={perf_metrics['accuracy']}")

    # Build monitoring record
    monitoring_record = {
        "snapshot_date":   snapshot_date,
        "model_name":      metadata["model_name"],
        "training_date":   metadata["training_date"],
        "n_predictions":   len(df_pred),
        "predicted_default_rate": round(float(df_pred["prediction"].mean()), 4),
        "psi":             psi,
        "psi_status":      interpret_psi(psi)[0] if psi is not None else None,
        **(perf_metrics or {}),
    }

    # Save monitoring record
    df_monitor = pd.DataFrame([monitoring_record])
    partition_name = snapshot_date.replace("-", "_") + ".parquet"
    out_path = os.path.join(gold_dir, "monitoring", partition_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_monitor.to_parquet(out_path, index=False)
    print(f"[Monitor] Saved monitoring record -> {out_path}")

    return monitoring_record


# Visualisation

def generate_monitoring_dashboard(gold_dir, output_dir):
    """
    Generate monitoring dashboard visualising:
    1. AUC over time - is the model still accurate?
    2. PSI over time - is the data distribution shifting month on month?
    3. Predicted vs actual default rate over time
    4. Score distribution for latest month (colour-coded by risk band)
    """
    monitor_files = sorted(glob.glob(os.path.join(gold_dir, "monitoring", "*.parquet")))
    if not monitor_files:
        print("[Visualise] No monitoring data found")
        return

    df_monitor = pd.concat([pd.read_parquet(f) for f in monitor_files], ignore_index=True)
    df_monitor = df_monitor.sort_values("snapshot_date").reset_index(drop=True)
    df_monitor["date"] = pd.to_datetime(df_monitor["snapshot_date"])

    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=0.95)

    BLUE   = "#1B4F8A"
    CYAN   = "#0891B2"
    RED    = "#DC2626"
    GREEN  = "#16A34A"
    ORANGE = "#EA580C"
    BG     = "#F8FAFC"
    TEXT   = "#1E293B"
    MUTED  = "#64748B"

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "ML Model Monitoring Dashboard\nLoan Default Prediction  |  CS611 Machine Learning Engineering",
        fontsize=15, fontweight="bold", y=0.99, color=TEXT,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    dates = df_monitor["date"]

    # Plot 1: AUC over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    has_auc = "auc" in df_monitor.columns and df_monitor["auc"].notna().any()
    if has_auc:
        auc_df = df_monitor[df_monitor["auc"].notna()]
        ax1.fill_between(auc_df["date"], auc_df["auc"], 0.5, alpha=0.12, color=BLUE)
        ax1.plot(auc_df["date"], auc_df["auc"],
                 marker="o", color=BLUE, linewidth=2.5, markersize=6, zorder=3)
        ax1.axhline(y=0.70, color=GREEN,  linestyle="--", linewidth=1.5, alpha=0.8,
                    label="Target (0.70)")
        ax1.axhline(y=0.65, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8,
                    label="Alert (0.65)")
        ax1.set_ylim(0.5, 1.0)
        ax1.legend(fontsize=8, loc="lower right")
        latest_auc = auc_df["auc"].iloc[-1]
        ax1.annotate(
            f"Latest: {latest_auc:.3f}",
            xy=(auc_df["date"].iloc[-1], latest_auc),
            xytext=(-60, 12), textcoords="offset points",
            fontsize=8, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=BLUE, alpha=0.5),
        )
    else:
        ax1.text(0.5, 0.5, "Labels not yet\navailable",
                 ha="center", va="center", transform=ax1.transAxes,
                 color=MUTED, fontsize=11)
    ax1.set_title("Model AUC Over Time", fontweight="bold", fontsize=12, pad=10, color=TEXT)
    ax1.set_xlabel("Snapshot Date", fontsize=9, color=MUTED)
    ax1.set_ylabel("AUC Score", fontsize=9, color=MUTED)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    # Plot 2: PSI bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    has_psi = "psi" in df_monitor.columns and df_monitor["psi"].notna().any()
    if has_psi:
        psi_vals = df_monitor["psi"].fillna(0)
        bar_colors = [GREEN if p < 0.10 else ORANGE if p < 0.25 else RED for p in psi_vals]
        x_pos = range(len(dates))
        ax2.bar(x_pos, psi_vals, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(
            [d.strftime("%Y-%m") for d in dates], rotation=45, ha="right", fontsize=7
        )
        ax2.axhline(y=0.10, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8)
        ax2.axhline(y=0.25, color=RED,    linestyle="--", linewidth=1.5, alpha=0.8)
        legend_patches = [
            Patch(facecolor=GREEN,  label="Stable  (PSI < 0.10)"),
            Patch(facecolor=ORANGE, label="Monitor (0.10 – 0.25)"),
            Patch(facecolor=RED,    label="Retrain (PSI ≥ 0.25)"),
        ]
        ax2.legend(handles=legend_patches, fontsize=7, loc="upper right")
    ax2.set_title("Population Stability Index (PSI)\nMonth-on-Month Comparison",
                  fontweight="bold", fontsize=12, pad=10, color=TEXT)
    ax2.set_xlabel("Snapshot Date", fontsize=9, color=MUTED)
    ax2.set_ylabel("PSI Value", fontsize=9, color=MUTED)
    ax2.tick_params(axis="y", labelsize=8)

    # Plot 3: Predicted vs Actual default rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    ax3.plot(dates, df_monitor["predicted_default_rate"],
             marker="o", color=CYAN, linewidth=2.5, markersize=6, label="Predicted")
    has_actual = (
        "default_rate_actual" in df_monitor.columns
        and df_monitor["default_rate_actual"].notna().any()
    )
    if has_actual:
        actual_series = df_monitor["default_rate_actual"].ffill()
        ax3.plot(dates, actual_series,
                 marker="s", color=RED, linewidth=2.5, markersize=6,
                 linestyle="--", label="Actual")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax3.set_title("Default Rate: Predicted vs Actual",
                  fontweight="bold", fontsize=12, pad=10, color=TEXT)
    ax3.set_xlabel("Snapshot Date", fontsize=9, color=MUTED)
    ax3.set_ylabel("Default Rate", fontsize=9, color=MUTED)
    ax3.tick_params(axis="x", rotation=45, labelsize=8)
    ax3.tick_params(axis="y", labelsize=8)

    # Plot 4: Score distribution (latest month, colour-coded by risk band)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    latest_pred_files = sorted(glob.glob(os.path.join(gold_dir, "predictions", "*.parquet")))
    if latest_pred_files:
        df_latest = pd.read_parquet(latest_pred_files[-1])
        latest_date = df_monitor["snapshot_date"].max()
        n, bins, patches_hist = ax4.hist(
            df_latest["probability"], bins=20, alpha=0.85,
            edgecolor="white", linewidth=0.5, color=BLUE,
        )
        for patch, left_edge in zip(patches_hist, bins[:-1]):
            patch.set_facecolor(GREEN if left_edge < 0.3 else ORANGE if left_edge < 0.7 else RED)
        ax4.axvline(x=0.5, color=RED, linestyle="--", linewidth=1.5, alpha=0.8,
                    label="Decision threshold (0.5)")
        ax4.legend(fontsize=8)
        ax4.set_title(f"Score Distribution\n(Latest snapshot: {latest_date})",
                      fontweight="bold", fontsize=12, pad=10, color=TEXT)
        ax4.set_xlabel("Predicted Default Probability", fontsize=9, color=MUTED)
        ax4.set_ylabel("Count", fontsize=9, color=MUTED)
        ax4.tick_params(axis="x", labelsize=8)
        ax4.tick_params(axis="y", labelsize=8)

    # Footer
    auc_rows = df_monitor[df_monitor["auc"].notna()] if "auc" in df_monitor.columns else pd.DataFrame()
    if not auc_rows.empty:
        last = auc_rows.iloc[-1]
        footer = (
            f"Model: {last.get('model_name', 'N/A')}  |  "
            f"Trained: {last.get('training_date', 'N/A')}  |  "
            f"AUC: {last.get('auc', 'N/A')}  |  "
            f"PSI: {last.get('psi', 'N/A')}  |  "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
    else:
        footer = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"

    fig.text(0.5, 0.005, footer, ha="center", fontsize=8, color=MUTED, style="italic")

    output_path = os.path.join(output_dir, "monitoring_dashboard.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Visualise] Dashboard saved as {output_path}")
    return output_path
