"""
=============================================================================
MEPS High-Cost Patient — Step 3b: Hyperparameter Tuning & Optimization
=============================================================================
Team 1 | MISM 6212

Prerequisites:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn

Input:
    ./features_output/train_features.csv
    ./features_output/test_features.csv

Output:
    ./tuned_output/tuning_results_summary.txt
    ./tuned_output/tuned_model_comparison.png
    ./tuned_output/tuned_lift_recall_at_k.png
    ./tuned_output/tuned_calibration.png
    ./tuned_output/threshold_analysis.png
    ./tuned_output/predictions_test_tuned.csv
    ./tuned_output/best_tuned_model.pkl

Usage:
    python step3b_tuning.py

Note:
    Full RandomizedSearchCV may take 10-30 minutes depending on hardware.
    Progress is printed for each model so you can track it.
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
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score,
    brier_score_loss, make_scorer, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.stats import uniform, randint, loguniform

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Skipping XGBoost tuning.\n")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR = "./features_output"
OUTPUT_DIR = "./tuned_output"
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER_SEARCH = 80          # number of random parameter combos per model
SCORING = "average_precision" # optimize for PR-AUC (best metric for imbalanced)

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
log("STEP 1: LOADING DATA")
log("=" * 72)

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_features.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test_features.csv"))

META_COLS = ["DUPERSID", "TOTEXP_NEXT", "HIGH_COST", "PERWT", "VARSTR", "VARPSU"]
feature_cols = [c for c in train_df.columns if c not in META_COLS]

X_train = train_df[feature_cols].values
y_train = train_df["HIGH_COST"].values
X_test = test_df[feature_cols].values
y_test = test_df["HIGH_COST"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log(f"Train: {X_train.shape}, Test: {X_test.shape}")
log(f"Features: {len(feature_cols)}")
log(f"Train prevalence: {y_train.mean()*100:.1f}%")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count
log(f"Class ratio (neg/pos): {scale_pos:.1f}:1")


# =============================================================================
# 2. HYPERPARAMETER SEARCH SPACES
# =============================================================================
log("\n" + "=" * 72)
log("STEP 2: HYPERPARAMETER TUNING (RandomizedSearchCV)")
log(f"  Scoring: {SCORING}")
log(f"  Iterations per model: {N_ITER_SEARCH}")
log(f"  CV folds: {CV_FOLDS}")
log("=" * 72)

search_spaces = {}

# --- Logistic Regression ---
search_spaces["Logistic Regression"] = {
    "estimator": LogisticRegression(
        max_iter=2000, solver="saga", random_state=RANDOM_STATE
    ),
    "params": {
        "C": loguniform(1e-3, 1e2),               # regularization strength
        "penalty": ["l1", "l2", "elasticnet"],
        "class_weight": ["balanced", {0: 1, 1: 10}, {0: 1, 1: 15}, {0: 1, 1: 20}],
        "l1_ratio": uniform(0, 1),                 # only used with elasticnet
    },
    "uses_scaled": True,
    "n_iter": N_ITER_SEARCH,
}

# --- Random Forest ---
search_spaces["Random Forest"] = {
    "estimator": RandomForestClassifier(
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "params": {
        "n_estimators": [300, 500, 800, 1000],
        "max_depth": [6, 8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [5, 10, 20, 30, 50],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": ["balanced", "balanced_subsample",
                         {0: 1, 1: 10}, {0: 1, 1: 15}],
    },
    "uses_scaled": False,
    "n_iter": N_ITER_SEARCH,
}

# --- XGBoost ---
if HAS_XGBOOST:
    search_spaces["XGBoost"] = {
        "estimator": XGBClassifier(
            eval_metric="aucpr",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "params": {
            "n_estimators": [300, 500, 800, 1000, 1500],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),          # 0.6 to 1.0
            "colsample_bytree": uniform(0.5, 0.5),   # 0.5 to 1.0
            "min_child_weight": [1, 3, 5, 7, 10],
            "gamma": loguniform(1e-3, 1.0),
            "reg_alpha": loguniform(1e-3, 10),
            "reg_lambda": loguniform(1e-3, 10),
            "scale_pos_weight": [scale_pos, scale_pos * 0.5,
                                 scale_pos * 0.75, scale_pos * 1.25],
        },
        "uses_scaled": False,
        "n_iter": N_ITER_SEARCH,
    }

# =============================================================================
# 3. RUN TUNING
# =============================================================================
tuned_models = {}

for name, spec in search_spaces.items():
    log(f"\n{'─'*60}")
    log(f"TUNING: {name}")
    log(f"{'─'*60}")
    t0 = time()

    X = X_train_scaled if spec["uses_scaled"] else X_train

    search = RandomizedSearchCV(
        estimator=spec["estimator"],
        param_distributions=spec["params"],
        n_iter=spec["n_iter"],
        scoring=SCORING,
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        return_train_score=True,
    )

    search.fit(X, y_train)
    elapsed = time() - t0

    log(f"  Best CV {SCORING}: {search.best_score_:.4f}")
    log(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"  Best params:")
    for k, v in search.best_params_.items():
        log(f"    {k}: {v}")

    # Store results
    tuned_models[name] = {
        "search": search,
        "best_model": search.best_estimator_,
        "best_cv_score": search.best_score_,
        "uses_scaled": spec["uses_scaled"],
    }

    # Show top 5 configs
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score").head(5)
    log(f"\n  Top 5 configurations:")
    for _, row in results_df.iterrows():
        train_s = row["mean_train_score"]
        test_s = row["mean_test_score"]
        std_s = row["std_test_score"]
        log(f"    CV={test_s:.4f} (±{std_s:.4f})  Train={train_s:.4f}  "
            f"Gap={train_s - test_s:.4f}")


# =============================================================================
# 4. CALIBRATION (Platt scaling via CalibratedClassifierCV)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 3: POST-HOC CALIBRATION (Platt Scaling)")
log("=" * 72)

calibrated_models = {}

for name, spec in tuned_models.items():
    log(f"\n  Calibrating {name}...")
    base_model = spec["best_model"]
    X = X_train_scaled if spec["uses_scaled"] else X_train

    # CalibratedClassifierCV with sigmoid (Platt) on cross-val folds
    cal_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=CV_FOLDS,
    )
    cal_model.fit(X, y_train)

    calibrated_models[name] = {
        "model": cal_model,
        "uses_scaled": spec["uses_scaled"],
    }
    log(f"    Done.")


# =============================================================================
# 5. EVALUATE ALL MODELS ON TEST SET
# =============================================================================
log("\n" + "=" * 72)
log("STEP 4: TEST SET EVALUATION (Tuned + Calibrated)")
log("=" * 72)

all_results = {}  # key: "ModelName (variant)" → results dict

for name, spec in tuned_models.items():
    for variant, model_obj, label in [
        ("tuned", spec["best_model"], f"{name} (tuned)"),
        ("calibrated", calibrated_models[name]["model"], f"{name} (tuned+calibrated)"),
    ]:
        X_te = X_test_scaled if spec["uses_scaled"] else X_test
        X_tr = X_train_scaled if spec["uses_scaled"] else X_train

        # Refit tuned (non-calibrated) on full train if needed
        if variant == "tuned":
            model_obj.fit(X_tr, y_train)

        y_proba = model_obj.predict_proba(X_te)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)

        all_results[label] = {
            "model": model_obj,
            "y_proba": y_proba,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "brier": brier,
            "uses_scaled": spec["uses_scaled"],
        }

        log(f"\n  {label}:")
        log(f"    ROC-AUC:   {roc_auc:.4f}")
        log(f"    PR-AUC:    {pr_auc:.4f}")
        log(f"    F1 @0.5:   {f1:.4f}")
        log(f"    Precision: {prec:.4f}")
        log(f"    Recall:    {rec:.4f}")
        log(f"    Brier:     {brier:.4f}")

        # Recall@K
        log(f"    Recall@K:")
        n_test = len(y_test)
        for k_pct in [1, 2, 5, 10, 15, 20]:
            k = max(1, int(n_test * k_pct / 100))
            top_k_idx = np.argsort(y_proba)[::-1][:k]
            captured = y_test[top_k_idx].sum()
            recall_k = captured / y_test.sum() * 100
            precision_k = captured / k * 100
            log(f"      Top {k_pct:>2}%: Recall={recall_k:5.1f}%, Precision={precision_k:5.1f}%")


# =============================================================================
# 6. THRESHOLD OPTIMIZATION
# =============================================================================
log("\n" + "=" * 72)
log("STEP 5: THRESHOLD OPTIMIZATION")
log("=" * 72)
log("  Finding optimal thresholds for different business objectives.\n")

# Pick the best calibrated model for threshold analysis
best_cal_name = max(
    [k for k in all_results if "calibrated" in k],
    key=lambda k: all_results[k]["pr_auc"]
)
best_proba = all_results[best_cal_name]["y_proba"]
log(f"  Using: {best_cal_name}")

# Strategy 1: Maximize F1
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, best_proba)
f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
best_f1_idx = np.argmax(f1_arr)
best_f1_thresh = thresh_arr[best_f1_idx]
log(f"\n  [Max F1] Threshold={best_f1_thresh:.3f}")
log(f"    F1={f1_arr[best_f1_idx]:.4f}, "
    f"Precision={prec_arr[best_f1_idx]:.4f}, "
    f"Recall={rec_arr[best_f1_idx]:.4f}")

# Strategy 2: Target recall >= 60%
recall_targets = [0.50, 0.60, 0.70, 0.80]
log(f"\n  [Target Recall] Thresholds:")
optimal_thresholds = {}
for target in recall_targets:
    # Find lowest threshold that achieves target recall
    valid = np.where(rec_arr[:-1] >= target)[0]
    if len(valid) > 0:
        # Among those achieving target recall, pick the one with highest precision
        best_idx = valid[np.argmax(prec_arr[:-1][valid])]
        t = thresh_arr[best_idx]
        y_pred_t = (best_proba >= t).astype(int)
        f1_t = f1_score(y_test, y_pred_t)
        prec_t = precision_score(y_test, y_pred_t, zero_division=0)
        rec_t = recall_score(y_test, y_pred_t)
        n_flagged = y_pred_t.sum()
        log(f"    Recall≥{target:.0%}: thresh={t:.3f}, Prec={prec_t:.3f}, "
            f"Rec={rec_t:.3f}, F1={f1_t:.3f}, Flagged={n_flagged}")
        optimal_thresholds[f"recall_{int(target*100)}"] = t

# Strategy 3: Capacity-constrained (flag top K patients)
log(f"\n  [Capacity-Constrained] Flag top N patients:")
for n_flag in [100, 200, 400, 600, 800]:
    if n_flag > len(y_test):
        continue
    top_idx = np.argsort(best_proba)[::-1][:n_flag]
    captured = y_test[top_idx].sum()
    total_hc = y_test.sum()
    log(f"    Top {n_flag:>4}: captures {captured:.0f}/{total_hc:.0f} "
        f"({captured/total_hc*100:.1f}%) high-cost, "
        f"precision={captured/n_flag*100:.1f}%")


# =============================================================================
# 7. COMPARISON SUMMARY TABLE
# =============================================================================
log("\n" + "=" * 72)
log("STEP 6: FINAL MODEL COMPARISON")
log("=" * 72)

# Only show calibrated versions (they should be used in production)
show_models = {k: v for k, v in all_results.items() if "calibrated" in k}

log(f"\n{'Model':<42s} {'ROC':>6s} {'PR':>6s} {'F1':>6s} {'Brier':>7s} {'R@5%':>6s} {'R@10%':>6s}")
log("─" * 82)
for label, res in show_models.items():
    # Recall@5% and @10%
    n = len(y_test)
    r5 = y_test[np.argsort(res["y_proba"])[::-1][:max(1, int(n*0.05))]].sum() / y_test.sum() * 100
    r10 = y_test[np.argsort(res["y_proba"])[::-1][:max(1, int(n*0.10))]].sum() / y_test.sum() * 100
    log(f"{label:<42s} {res['roc_auc']:>6.3f} {res['pr_auc']:>6.3f} "
        f"{res['f1']:>6.3f} {res['brier']:>7.4f} {r5:>5.1f}% {r10:>5.1f}%")

overall_best = max(show_models, key=lambda k: show_models[k]["pr_auc"])
log(f"\nBest model (by PR-AUC): {overall_best}")
log(f"  PR-AUC improvement over Step 3 baseline:")
log(f"  (compare these numbers to the Step 3 results to see gains)")


# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 7: GENERATING VISUALIZATIONS")
log("=" * 72)

model_colors = {
    "Logistic Regression (tuned+calibrated)": "#1565C0",
    "Random Forest (tuned+calibrated)": "#43A047",
    "XGBoost (tuned+calibrated)": "#E53935",
}
# Fallback colors for any model name
def get_color(name):
    for k, v in model_colors.items():
        if k in name:
            return v
    return "gray"


# ---- FIGURE 1: ROC + PR Curves (calibrated models only) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Tuned + Calibrated Model Performance — Temporal Validation",
             fontsize=13, fontweight="bold")

ax = axes[0]
for label, res in show_models.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f'{label}\n(AUC={res["roc_auc"]:.3f})',
            color=get_color(label), lw=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right", fontsize=8)

ax = axes[1]
baseline = y_test.mean()
for label, res in show_models.items():
    p, r, _ = precision_recall_curve(y_test, res["y_proba"])
    ax.plot(r, p, label=f'{label}\n(AP={res["pr_auc"]:.3f})',
            color=get_color(label), lw=2)
ax.axhline(baseline, color="gray", ls="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "tuned_model_comparison.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved tuned_model_comparison.png")


# ---- FIGURE 2: Recall@K + Lift ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Targeting Efficiency — Tuned + Calibrated Models", fontsize=13, fontweight="bold")

k_pcts = np.arange(1, 31)

ax = axes[0]
for label, res in show_models.items():
    recalls = []
    for k_pct in k_pcts:
        k = max(1, int(len(y_test) * k_pct / 100))
        top_k = np.argsort(res["y_proba"])[::-1][:k]
        recalls.append(y_test[top_k].sum() / y_test.sum() * 100)
    ax.plot(k_pcts, recalls, label=label.replace(" (tuned+calibrated)", ""),
            color=get_color(label), lw=2, marker="o", ms=3)
ax.plot(k_pcts, k_pcts, "k--", alpha=0.4, label="Random")
ax.set_xlabel("Top K% Targeted")
ax.set_ylabel("% of True High-Cost Captured")
ax.set_title("Recall@K")
ax.legend(fontsize=8)
ax.set_xlim(1, 30)
ax.set_ylim(0, 100)

ax = axes[1]
for label, res in show_models.items():
    lifts = []
    for k_pct in k_pcts:
        k = max(1, int(len(y_test) * k_pct / 100))
        top_k = np.argsort(res["y_proba"])[::-1][:k]
        lift = y_test[top_k].mean() / y_test.mean()
        lifts.append(lift)
    ax.plot(k_pcts, lifts, label=label.replace(" (tuned+calibrated)", ""),
            color=get_color(label), lw=2, marker="o", ms=3)
ax.axhline(1, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Top K% Targeted")
ax.set_ylabel("Lift over Random")
ax.set_title("Lift Curve")
ax.legend(fontsize=8)
ax.set_xlim(1, 30)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "tuned_lift_recall_at_k.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved tuned_lift_recall_at_k.png")


# ---- FIGURE 3: Calibration Curves ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Calibration: Before vs After Platt Scaling", fontsize=13, fontweight="bold")

# Before calibration
ax = axes[0]
ax.set_title("Tuned (before calibration)")
for label, res in all_results.items():
    if "calibrated" in label:
        continue
    prob_true, prob_pred = calibration_curve(y_test, res["y_proba"], n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", label=f'{label}\n(Brier={res["brier"]:.4f})',
            color=get_color(label), lw=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.legend(loc="lower right", fontsize=8)

# After calibration
ax = axes[1]
ax.set_title("Tuned + Calibrated (Platt scaling)")
for label, res in show_models.items():
    prob_true, prob_pred = calibration_curve(y_test, res["y_proba"], n_bins=10, strategy="uniform")
    ax.plot(prob_pred, prob_true, marker="o", label=f'{label}\n(Brier={res["brier"]:.4f})',
            color=get_color(label), lw=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "tuned_calibration.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved tuned_calibration.png")


# ---- FIGURE 4: Threshold Analysis ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Threshold Analysis — {best_cal_name}", fontsize=13, fontweight="bold")

# 4a: Precision-Recall-F1 vs threshold
ax = axes[0]
thresholds_plot = np.linspace(0.01, 0.5, 200)
precs, recs, f1s = [], [], []
for t in thresholds_plot:
    y_p = (best_proba >= t).astype(int)
    precs.append(precision_score(y_test, y_p, zero_division=0))
    recs.append(recall_score(y_test, y_p))
    f1s.append(f1_score(y_test, y_p))

ax.plot(thresholds_plot, precs, label="Precision", color="#1565C0", lw=2)
ax.plot(thresholds_plot, recs, label="Recall", color="#E53935", lw=2)
ax.plot(thresholds_plot, f1s, label="F1", color="#FF8F00", lw=2, ls="--")
ax.axvline(best_f1_thresh, color="gray", ls=":", alpha=0.7, label=f"Max F1 ({best_f1_thresh:.3f})")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs Threshold")
ax.legend()

# 4b: Number flagged vs recall captured
ax = axes[1]
n_flagged_arr = []
recall_arr = []
for t in np.linspace(0.01, 0.99, 200):
    y_p = (best_proba >= t).astype(int)
    n_flagged_arr.append(y_p.sum())
    recall_arr.append(recall_score(y_test, y_p))

ax.plot(n_flagged_arr, [r * 100 for r in recall_arr], color="#43A047", lw=2)
ax.set_xlabel("Number of Patients Flagged")
ax.set_ylabel("% of True High-Cost Captured (Recall)")
ax.set_title("Operational Curve: Flagged Patients vs Capture Rate")
# Annotate key points
for n_target in [200, 400, 800]:
    diffs = [abs(n - n_target) for n in n_flagged_arr]
    idx = np.argmin(diffs)
    ax.annotate(f"n={n_flagged_arr[idx]}: {recall_arr[idx]*100:.0f}%",
                xy=(n_flagged_arr[idx], recall_arr[idx] * 100),
                fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.plot(n_flagged_arr[idx], recall_arr[idx] * 100, "ro", ms=6)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "threshold_analysis.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved threshold_analysis.png")


# =============================================================================
# 9. SAVE OUTPUTS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 8: SAVING OUTPUTS")
log("=" * 72)

# Predictions with all model probabilities
pred_df = test_df[["DUPERSID", "HIGH_COST", "TOTEXP_NEXT"]].copy()
for label, res in all_results.items():
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").lower()
    pred_df[f"proba_{safe}"] = res["y_proba"]
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_test_tuned.csv"), index=False)
log("  Saved predictions_test_tuned.csv")

# Best model pickle
best_res = all_results[overall_best]
save_obj = {
    "model": best_res["model"],
    "model_name": overall_best,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "optimal_thresholds": optimal_thresholds,
    "best_f1_threshold": best_f1_thresh,
    "test_metrics": {
        "roc_auc": best_res["roc_auc"],
        "pr_auc": best_res["pr_auc"],
        "brier": best_res["brier"],
    },
}
with open(os.path.join(OUTPUT_DIR, "best_tuned_model.pkl"), "wb") as f:
    pickle.dump(save_obj, f)
log(f"  Saved best_tuned_model.pkl ({overall_best})")

# Summary text
with open(os.path.join(OUTPUT_DIR, "tuning_results_summary.txt"), "w") as f:
    f.write("\n".join(log_lines))
log("  Saved tuning_results_summary.txt")

log(f"\n{'='*72}")
log(f"ALL DONE! Outputs in {OUTPUT_DIR}/")
log(f"Best model: {overall_best}")
log(f"  PR-AUC = {best_res['pr_auc']:.4f}")
log(f"  ROC-AUC = {best_res['roc_auc']:.4f}")
log(f"  Brier = {best_res['brier']:.4f}")
log(f"{'='*72}")