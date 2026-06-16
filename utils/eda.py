"""
EDA Utility:
Exploratory Data Analysis on the gold feature store and label store.
Runs at each training cycle and saves a report to reports/eda_report.png.

"""

import os
import glob

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


def run_eda(gold_dir, reports_dir, training_date):
    print(f"[EDA] Running EDA for training_date={training_date}")

    feature_files = sorted(glob.glob(os.path.join(gold_dir, "feature_store", "*.parquet")))
    label_files   = sorted(glob.glob(os.path.join(gold_dir, "label_store",   "*.parquet")))

    # Keep only files up to training_date
    def in_range(f):
        name = os.path.basename(f).replace(".parquet", "").replace("_", "-")
        return name <= training_date

    feature_files = [f for f in feature_files if in_range(f)]
    label_files   = [f for f in label_files   if in_range(f)]

    if not feature_files or not label_files:
        print("[EDA] Not enough data yet, skipping")
        return None

    df_feat  = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)
    df_label = pd.concat([pd.read_parquet(f) for f in label_files],   ignore_index=True)

    df = df_feat.merge(df_label[["customer_id", "label"]], on="customer_id", how="inner")
    df = df.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)

    n_total    = len(df)
    n_default  = int(df["label"].sum())
    default_rt = df["label"].mean()
    print(f"[EDA] Rows={n_total} | Defaults={n_default} ({default_rt:.2%})")

    # Drop ID / date columns for analysis
    drop_cols  = ["customer_id", "loan_id", "snapshot_date", "loan_start_date", "label_def"]
    num_df     = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    num_df     = num_df.select_dtypes(include=[np.number])

    os.makedirs(reports_dir, exist_ok=True)

    # Palette 
    sns.set_theme(style="whitegrid", font_scale=0.9)
    BLUE   = "#1B4F8A"
    RED    = "#DC2626"
    GREEN  = "#16A34A"
    ORANGE = "#EA580C"
    BG     = "#F8FAFC"
    TEXT   = "#1E293B"
    MUTED  = "#64748B"

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Exploratory Data Analysis  |  Loan Default Prediction\n"
        f"Training snapshot up to {training_date}  |  {n_total:,} records",
        fontsize=14, fontweight="bold", y=0.99, color=TEXT,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

    # Panel 1: Class distribution 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    counts = df["label"].value_counts().sort_index()
    bars = ax1.bar(["No Default (0)", "Default (1)"], counts.values,
                   color=[GREEN, RED], alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + n_total * 0.01,
                 f"{val:,}\n({val/n_total:.1%})", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=TEXT)
    ax1.set_title("Class Distribution", fontweight="bold", fontsize=12, color=TEXT)
    ax1.set_ylabel("Count", fontsize=9, color=MUTED)
    ax1.tick_params(labelsize=9)
    ax1.set_ylim(0, counts.max() * 1.18)
    imbalance_ratio = counts[0] / counts[1] if counts[1] > 0 else 0
    ax1.text(0.98, 0.95, f"Imbalance ratio\n{imbalance_ratio:.1f} : 1",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=9, color=MUTED, style="italic")

    # Panel 2: Missing values (top 15 features) 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    missing_pct = (num_df.isnull().mean() * 100).sort_values(ascending=False).head(15)
    if missing_pct.sum() > 0:
        colors_miss = [RED if v > 40 else ORANGE if v > 10 else BLUE for v in missing_pct]
        ax2.barh(missing_pct.index, missing_pct.values,
                 color=colors_miss, alpha=0.85, edgecolor="white")
        ax2.axvline(x=40, color=RED,    linestyle="--", linewidth=1.2, alpha=0.7, label="Drop threshold (40%)")
        ax2.axvline(x=10, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.7, label="Flag threshold (10%)")
        ax2.legend(fontsize=8)
        ax2.set_xlabel("Missing %", fontsize=9, color=MUTED)
    else:
        ax2.text(0.5, 0.5, "No missing values", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11, color=MUTED)
    ax2.set_title("Missing Values by Feature (Top 15)", fontweight="bold", fontsize=12, color=TEXT)
    ax2.tick_params(labelsize=8)

    # Panel 3: Feature correlations with label 
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    corr_with_label = (
        num_df.fillna(0)
        .corrwith(df["label"])
        .abs()
        .dropna()
        .sort_values(ascending=False)
        .head(15)
    )
    bar_colors = [BLUE if v >= 0 else RED
                  for v in num_df.fillna(0).corrwith(df["label"]).reindex(corr_with_label.index)]
    ax3.barh(corr_with_label.index[::-1], corr_with_label.values[::-1],
             color=bar_colors[::-1], alpha=0.85, edgecolor="white")
    ax3.set_xlabel("|Correlation| with Label", fontsize=9, color=MUTED)
    ax3.set_title("Top 15 Feature Correlations with Target", fontweight="bold", fontsize=12, color=TEXT)
    ax3.tick_params(labelsize=8)

    # Panel 4: Key features distribution by default status
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    key_features = [c for c in ["outstanding_debt", "credit_utilization_ratio",
                                 "delay_from_due_date", "annual_income"]
                    if c in num_df.columns]
    if key_features:
        feat = key_features[0]   # pick the most relevant
        defaults     = df[df["label"] == 1][feat].dropna()
        non_defaults = df[df["label"] == 0][feat].dropna()
        ax4.hist(non_defaults, bins=40, alpha=0.6, color=GREEN, label="No Default", density=True)
        ax4.hist(defaults,     bins=40, alpha=0.6, color=RED,   label="Default",    density=True)
        ax4.legend(fontsize=9)
        ax4.set_xlabel(feat.replace("_", " ").title(), fontsize=9, color=MUTED)
        ax4.set_ylabel("Density", fontsize=9, color=MUTED)
    ax4.set_title("Feature Distribution by Default Status", fontweight="bold", fontsize=12, color=TEXT)
    ax4.tick_params(labelsize=8)

    # Panel 5: Default rate by credit mix 
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(BG)
    credit_mix_cols = [c for c in df.columns if c.startswith("credit_mix_")]
    if credit_mix_cols:
        mix_rates = {}
        for col in credit_mix_cols:
            mask = df[col] == 1
            if mask.sum() > 0:
                label = col.replace("credit_mix_", "").title()
                mix_rates[label] = df.loc[mask, "label"].mean()
        if mix_rates:
            labels_cm = list(mix_rates.keys())
            rates_cm  = list(mix_rates.values())
            bar_c = [GREEN if r < default_rt else RED for r in rates_cm]
            ax5.bar(labels_cm, rates_cm, color=bar_c, alpha=0.85, edgecolor="white", width=0.5)
            ax5.axhline(y=default_rt, color=BLUE, linestyle="--", linewidth=1.5,
                        alpha=0.8, label=f"Overall ({default_rt:.1%})")
            ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax5.legend(fontsize=8)
    ax5.set_title("Default Rate by Credit Mix", fontweight="bold", fontsize=12, color=TEXT)
    ax5.set_ylabel("Default Rate", fontsize=9, color=MUTED)
    ax5.tick_params(labelsize=9)

    # Panel 6: Feature importance (quick RF) 
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(BG)
    try:
        X_imp = num_df.drop(columns=["label"], errors="ignore").fillna(0)
        y_imp = df["label"]
        rf_quick = RandomForestClassifier(n_estimators=50, max_depth=6,
                                          random_state=42, n_jobs=-1)
        rf_quick.fit(X_imp, y_imp)
        importances = pd.Series(rf_quick.feature_importances_, index=X_imp.columns)
        top_imp = importances.nlargest(15).sort_values()
        ax6.barh(top_imp.index, top_imp.values,
                 color=BLUE, alpha=0.85, edgecolor="white")
        ax6.set_xlabel("Feature Importance", fontsize=9, color=MUTED)
        ax6.set_title("Top 15 Features by RF Importance", fontweight="bold", fontsize=12, color=TEXT)
        ax6.tick_params(labelsize=8)
    except Exception as e:
        ax6.text(0.5, 0.5, f"Could not compute\n{e}", ha="center", va="center",
                 transform=ax6.transAxes, color=MUTED, fontsize=9)

    # Footer 
    fig.text(0.5, 0.005,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  |  "
             f"Records: {n_total:,}  |  Features: {num_df.shape[1]}  |  "
             f"Default rate: {default_rt:.2%}",
             ha="center", fontsize=8, color=MUTED, style="italic")

    output_path = os.path.join(reports_dir, "eda_report.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[EDA] Report saved → {output_path}")

    return {
        "n_total":           n_total,
        "n_default":         n_default,
        "default_rate":      round(float(default_rt), 4),
        "imbalance_ratio":   round(float(imbalance_ratio), 2),
        "n_features":        num_df.shape[1],
        "top_feature":       corr_with_label.index[0] if len(corr_with_label) else None,
    }
