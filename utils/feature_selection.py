"""
Feature Selection Utility:

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def remove_low_variance(X, threshold=0.01):
    """Drop columns whose variance is below threshold (near-constant features)."""
    variances = X.var()
    keep = variances[variances > threshold].index.tolist()
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print(f"[FeatSelect] Step 1 - removed {len(dropped)} low-variance feature(s): "
              f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}")
    else:
        print(f"[FeatSelect] Step 1 - no low-variance features found")
    return X[keep], keep


def remove_highly_correlated(X, threshold=0.85):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    keep = [c for c in X.columns if c not in to_drop]
    if to_drop:
        print(f"[FeatSelect] Step 2 - removed {len(to_drop)} highly correlated feature(s) "
              f"(threshold={threshold})")
    else:
        print(f"[FeatSelect] Step 2 - no highly correlated features found")
    return X[keep], keep


def select_by_rf_importance(X, y, top_n=40):
    n_select = min(top_n, X.shape[1])
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=6,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(n_select).index.tolist()
    print(f"[FeatSelect] Step 3 - kept top {len(top_features)} features by RF importance")
    return X[top_features], top_features, importances.sort_values(ascending=False)


def select_features(X_train, y_train, top_n=40):
    print(f"[FeatSelect] Starting with {X_train.shape[1]} features, "
          f"{X_train.shape[0]} samples")

    X_train, cols = remove_low_variance(X_train)
    X_train, cols = remove_highly_correlated(X_train)
    X_train, cols, importances = select_by_rf_importance(X_train, y_train, top_n=top_n)

    print(f"[FeatSelect] Final selection: {len(cols)} features")
    print(f"[FeatSelect] Top 5: {cols[:5]}")
    return X_train, cols, importances
