"""
=============================================================================
MEPS High-Cost Patient — Step 3: Modeling Pipeline
=============================================================================
Team 1 | MISM 6212

Prerequisites:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn

Input:
    ./features_output/train_features.csv
    ./features_output/test_features.csv

Output:
    ./model_output/model_comparison.png         (ROC + PR curves)
    ./model_output/lift_recall_at_k.png         (Recall@K & Lift curves)
    ./model_output/feature_importance.png       (top features per model)
    ./model_output/calibration_curves.png       (reliability diagrams)
    ./model_output/model_results_summary.txt    (full metrics text)
    ./model_output/predictions_test.csv         (test set predictions)
    ./model_output/best_model.pkl               (serialized best model)

Usage:
    python step3_modeling.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Install with: pip install xgboost")
    print("         Proceeding with Logistic Regression and Random Forest only.\n")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR = "./features_output"
OUTPUT_DIR = "./model_output"
RANDOM_STATE = 42
CV_FOLDS = 5
HIGH_COST_PERCENTILE = 0.05  # top 5%

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "figure.facecolor": "white"})
sns.set_style("whitegrid")

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(text)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
log("=" * 72)
log("STEP 1: LOADING FEATURE-ENGINEERED DATA")
log("=" * 72)

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_features.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test_features.csv"))

log(f"Train: {train_df.shape}")
log(f"Test:  {test_df.shape}")

# Separate features, target, metadata
META_COLS = ["DUPERSID", "TOTEXP_NEXT", "HIGH_COST", "PERWT", "VARSTR", "VARPSU"]
feature_cols = [c for c in train_df.columns if c not in META_COLS]

X_train = train_df[feature_cols].values
y_train = train_df["HIGH_COST"].values
X_test = test_df[feature_cols].values
y_test = test_df["HIGH_COST"].values

log(f"Features: {len(feature_cols)}")
log(f"Train target: {np.bincount(y_train)} (0s, 1s)")
log(f"Test target:  {np.bincount(y_test)} (0s, 1s)")
log(f"Train prevalence: {y_train.mean()*100:.1f}%")
log(f"Test prevalence:  {y_test.mean()*100:.1f}%")


# =============================================================================
# 2. SCALE FEATURES (needed for Logistic Regression)
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# 3. DEFINE MODELS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 2: MODEL DEFINITIONS")
log("=" * 72)

# Class weight to handle imbalance
# Ratio: ~19:1 (non-HC:HC), so we upweight the minority class
models = {}

# --- Logistic Regression ---
models["Logistic Regression"] = {
    "model": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),
    "uses_scaled": True,  # LR benefits from scaling
}

# --- Random Forest ---
models["Random Forest"] = {
    "model": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "uses_scaled": False,  # trees don't need scaling
}

# --- XGBoost ---
if HAS_XGBOOST:
    # Calculate scale_pos_weight for imbalanced target
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count

    models["XGBoost"] = {
        "model": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "uses_scaled": False,
    }

for name in models:
    log(f"  {name}: {models[name]['model'].__class__.__name__}")


# =============================================================================
# 4. CROSS-VALIDATION ON TRAIN SET
# =============================================================================
log("\n" + "=" * 72)
log("STEP 3: 5-FOLD STRATIFIED CROSS-VALIDATION (Train Set)")
log("=" * 72)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, spec in models.items():
    log(f"\n--- {name} ---")
    t0 = time()

    model = spec["model"]
    X = X_train_scaled if spec["uses_scaled"] else X_train

    # Get out-of-fold predicted probabilities
    y_proba_cv = cross_val_predict(model, X, y_train, cv=cv, method="predict_proba")[:, 1]

    roc_auc = roc_auc_score(y_train, y_proba_cv)
    pr_auc = average_precision_score(y_train, y_proba_cv)

    # Threshold at 0.5
    y_pred_cv = (y_proba_cv >= 0.5).astype(int)
    f1 = f1_score(y_train, y_pred_cv)
    prec = precision_score(y_train, y_pred_cv, zero_division=0)
    rec = recall_score(y_train, y_pred_cv)

    elapsed = time() - t0
    log(f"  ROC-AUC:  {roc_auc:.4f}")
    log(f"  PR-AUC:   {pr_auc:.4f}")
    log(f"  F1 @0.5:  {f1:.4f}")
    log(f"  Prec @0.5:{prec:.4f}")
    log(f"  Recall@0.5:{rec:.4f}")
    log(f"  Time: {elapsed:.1f}s")

    cv_results[name] = {
        "y_proba_cv": y_proba_cv,
        "roc_auc_cv": roc_auc,
        "pr_auc_cv": pr_auc,
        "f1_cv": f1,
    }


# =============================================================================
# 5. FIT ON FULL TRAIN, EVALUATE ON TEST (TEMPORAL VALIDATION)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 4: TEMPORAL VALIDATION (Train on 2021→2022, Test on 2022→2023)")
log("=" * 72)

test_results = {}

for name, spec in models.items():
    log(f"\n--- {name} ---")
    t0 = time()

    model = spec["model"]
    X_tr = X_train_scaled if spec["uses_scaled"] else X_train
    X_te = X_test_scaled if spec["uses_scaled"] else X_test

    # Fit on full training set
    model.fit(X_tr, y_train)

    # Predict on test
    y_proba_test = model.predict_proba(X_te)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba_test)
    pr_auc = average_precision_score(y_test, y_proba_test)
    f1 = f1_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test)
    brier = brier_score_loss(y_test, y_proba_test)

    elapsed = time() - t0

    log(f"  ROC-AUC:   {roc_auc:.4f}")
    log(f"  PR-AUC:    {pr_auc:.4f}")
    log(f"  F1 @0.5:   {f1:.4f}")
    log(f"  Precision: {prec:.4f}")
    log(f"  Recall:    {rec:.4f}")
    log(f"  Brier:     {brier:.4f}")
    log(f"  Time: {elapsed:.1f}s")

    # Recall@K analysis (key business metric)
    log(f"\n  Recall@K (what % of true high-cost captured when targeting top K%):")
    n_test = len(y_test)
    for k_pct in [1, 2, 5, 10, 15, 20]:
        k = max(1, int(n_test * k_pct / 100))
        top_k_idx = np.argsort(y_proba_test)[::-1][:k]
        captured = y_test[top_k_idx].sum()
        total_hc = y_test.sum()
        recall_k = captured / total_hc * 100
        precision_k = captured / k * 100
        log(f"    Top {k_pct:>2}% (n={k:>4}): Recall={recall_k:5.1f}%, Precision={precision_k:5.1f}%")

    # Confusion matrix at 0.5 threshold
    cm = confusion_matrix(y_test, y_pred_test)
    log(f"\n  Confusion Matrix (threshold=0.5):")
    log(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    log(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    test_results[name] = {
        "model": model,
        "y_proba_test": y_proba_test,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "brier": brier,
    }


# =============================================================================
# 6. COMPARISON SUMMARY TABLE
# =============================================================================
log("\n" + "=" * 72)
log("STEP 5: MODEL COMPARISON SUMMARY")
log("=" * 72)

log(f"\n{'Model':<22s} {'CV ROC':>8s} {'CV PR':>8s} {'Test ROC':>9s} {'Test PR':>8s} {'Test F1':>8s} {'Brier':>7s}")
log("-" * 72)
for name in models:
    cv_r = cv_results[name]
    te_r = test_results[name]
    log(f"{name:<22s} {cv_r['roc_auc_cv']:>8.4f} {cv_r['pr_auc_cv']:>8.4f} "
        f"{te_r['roc_auc']:>9.4f} {te_r['pr_auc']:>8.4f} {te_r['f1']:>8.4f} {te_r['brier']:>7.4f}")

# Identify best model by test PR-AUC (most relevant for imbalanced problems)
best_name = max(test_results, key=lambda k: test_results[k]["pr_auc"])
log(f"\nBest model (by Test PR-AUC): {best_name}")


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 6: GENERATING VISUALIZATIONS")
log("=" * 72)

model_colors = {
    "Logistic Regression": "#1565C0",
    "Random Forest": "#43A047",
    "XGBoost": "#E53935",
}


# ---- FIGURE 1: ROC + PR Curves ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Performance — Temporal Validation (Train: 2021→2022, Test: 2022→2023)",
             fontsize=13, fontweight="bold")

# ROC
ax = axes[0]
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba_test"])
    ax.plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})',
            color=model_colors.get(name, "gray"), lw=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")

# PR
ax = axes[1]
baseline_pr = y_test.mean()
for name, res in test_results.items():
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, res["y_proba_test"])
    ax.plot(rec_arr, prec_arr, label=f'{name} (AP={res["pr_auc"]:.3f})',
            color=model_colors.get(name, "gray"), lw=2)
ax.axhline(baseline_pr, color="gray", ls="--", alpha=0.5, label=f"Baseline ({baseline_pr:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend(loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved model_comparison.png")


# ---- FIGURE 2: Recall@K and Lift Curves ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Targeting Efficiency — Recall@K and Lift Curves",
             fontsize=13, fontweight="bold")

k_pcts = np.arange(1, 31)

# Recall@K
ax = axes[0]
for name, res in test_results.items():
    recalls = []
    for k_pct in k_pcts:
        k = max(1, int(len(y_test) * k_pct / 100))
        top_k_idx = np.argsort(res["y_proba_test"])[::-1][:k]
        recall_k = y_test[top_k_idx].sum() / y_test.sum() * 100
        recalls.append(recall_k)
    ax.plot(k_pcts, recalls, label=name, color=model_colors.get(name, "gray"), lw=2, marker="o", ms=3)

# Random baseline
random_recalls = [k for k in k_pcts]
ax.plot(k_pcts, random_recalls, "k--", alpha=0.4, label="Random")
ax.set_xlabel("Top K% Targeted")
ax.set_ylabel("% of True High-Cost Captured (Recall)")
ax.set_title("Recall@K — Targeting Efficiency")
ax.legend()
ax.set_xlim(1, 30)
ax.set_ylim(0, 100)

# Lift
ax = axes[1]
for name, res in test_results.items():
    lifts = []
    for k_pct in k_pcts:
        k = max(1, int(len(y_test) * k_pct / 100))
        top_k_idx = np.argsort(res["y_proba_test"])[::-1][:k]
        precision_k = y_test[top_k_idx].mean()
        lift = precision_k / y_test.mean()
        lifts.append(lift)
    ax.plot(k_pcts, lifts, label=name, color=model_colors.get(name, "gray"), lw=2, marker="o", ms=3)

ax.axhline(1, color="gray", ls="--", alpha=0.5, label="No lift (random)")
ax.set_xlabel("Top K% Targeted")
ax.set_ylabel("Lift over Random")
ax.set_title("Lift Curve — Precision Gain")
ax.legend()
ax.set_xlim(1, 30)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "lift_recall_at_k.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved lift_recall_at_k.png")


# ---- FIGURE 3: Feature Importance ----
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 8))
if len(models) == 1:
    axes = [axes]
fig.suptitle("Top 20 Features by Model", fontsize=13, fontweight="bold")

for idx, (name, res) in enumerate(test_results.items()):
    ax = axes[idx]
    model = res["model"]

    if name == "Logistic Regression":
        importances = np.abs(model.coef_[0])
    elif name == "Random Forest":
        importances = model.feature_importances_
    elif name == "XGBoost":
        importances = model.feature_importances_
    else:
        continue

    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    top20 = feat_imp.tail(20)
    top20.plot(kind="barh", ax=ax, color=model_colors.get(name, "gray"), alpha=0.8)
    ax.set_title(name)
    ax.set_xlabel("Importance" if name != "Logistic Regression" else "|Coefficient|")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved feature_importance.png")


# ---- FIGURE 4: Calibration Curves ----
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
ax.set_title("Calibration Curves (Reliability Diagram)", fontsize=13, fontweight="bold")

for name, res in test_results.items():
    prob_true, prob_pred = calibration_curve(y_test, res["y_proba_test"], n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", label=name,
            color=model_colors.get(name, "gray"), lw=2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.legend(loc="lower right")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "calibration_curves.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved calibration_curves.png")


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 7: SAVING OUTPUTS")
log("=" * 72)

# Save test predictions
pred_df = test_df[["DUPERSID", "HIGH_COST", "TOTEXP_NEXT"]].copy()
for name, res in test_results.items():
    safe_name = name.replace(" ", "_").lower()
    pred_df[f"proba_{safe_name}"] = res["y_proba_test"]
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_test.csv"), index=False)
log("  Saved predictions_test.csv")

# Save best model
best_model = test_results[best_name]["model"]
with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump({"model": best_model, "scaler": scaler, "feature_cols": feature_cols,
                 "model_name": best_name}, f)
log(f"  Saved best_model.pkl ({best_name})")

# Save summary text
summary_path = os.path.join(OUTPUT_DIR, "model_results_summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(log_lines))
log("  Saved model_results_summary.txt")

log(f"\n{'='*72}")
log(f"ALL DONE! Outputs saved to {OUTPUT_DIR}/")
log(f"Best model: {best_name} (Test PR-AUC = {test_results[best_name]['pr_auc']:.4f})")
log(f"{'='*72}")