"""
=============================================================================
MEPS High-Cost Patient — Step 4: SHAP Explainability
=============================================================================
Team 1 | MISM 6212

Prerequisites:
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

Input:
    ./features_output/train_features.csv
    ./features_output/test_features.csv
    ./tuned_output/best_tuned_model.pkl

Output:
    ./shap_output/shap_summary_bar.png          (global feature importance)
    ./shap_output/shap_summary_beeswarm.png     (direction of feature effects)
    ./shap_output/shap_dependence_top6.png      (partial dependence for top features)
    ./shap_output/shap_force_high_cost.png      (example high-cost patient explanation)
    ./shap_output/shap_force_low_cost.png       (example low-cost patient explanation)
    ./shap_output/shap_interaction_top.png       (top interaction effects)
    ./shap_output/shap_group_importance.png      (importance by feature group)
    ./shap_output/shap_values.csv               (SHAP values for all test observations)
    ./shap_output/shap_analysis_summary.txt     (text summary of findings)

Usage:
    python step4_shap.py

Note:
    SHAP TreeExplainer is fast for XGBoost (~1-3 min).
    If you used a different best model, the script auto-detects and adjusts.
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
from collections import OrderedDict

warnings.filterwarnings("ignore")

# Suppress SHAP's tqdm bars in non-interactive mode
os.environ["TQDM_DISABLE"] = "1"

import shap

# =============================================================================
# CONFIG
# =============================================================================
FEATURES_DIR = "./features_output"
TUNED_DIR = "./tuned_output"
OUTPUT_DIR = "./shap_output"
RANDOM_STATE = 42
TOP_N_FEATURES = 20  # how many features to show in plots

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "figure.facecolor": "white"})
sns.set_style("whitegrid")

log_lines = []
def log(text=""):
    print(text)
    log_lines.append(text)


# =============================================================================
# 1. LOAD DATA & MODEL
# =============================================================================
log("=" * 72)
log("STEP 1: LOADING DATA & MODEL")
log("=" * 72)

train_df = pd.read_csv(os.path.join(FEATURES_DIR, "train_features.csv"))
test_df = pd.read_csv(os.path.join(FEATURES_DIR, "test_features.csv"))

with open(os.path.join(TUNED_DIR, "best_tuned_model.pkl"), "rb") as f:
    model_pkg = pickle.load(f)

model = model_pkg["model"]
model_name = model_pkg["model_name"]
feature_cols = model_pkg["feature_cols"]
scaler = model_pkg["scaler"]

log(f"Model: {model_name}")
log(f"Features: {len(feature_cols)}")

META_COLS = ["DUPERSID", "TOTEXP_NEXT", "HIGH_COST", "PERWT", "VARSTR", "VARPSU"]
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_test = test_df["HIGH_COST"].values

# Determine if model needs scaled data
needs_scaling = "logistic" in model_name.lower()
if needs_scaling:
    X_train_use = scaler.transform(X_train)
    X_test_use = scaler.transform(X_test)
    log("  Using scaled features (Logistic Regression detected)")
else:
    X_train_use = X_train
    X_test_use = X_test
    log("  Using raw features (tree-based model)")

# Create DataFrames with feature names for SHAP plots
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
X_train_df = pd.DataFrame(X_train, columns=feature_cols)


# =============================================================================
# 2. COMPUTE SHAP VALUES
# =============================================================================
log("\n" + "=" * 72)
log("STEP 2: COMPUTING SHAP VALUES")
log("=" * 72)

# For CalibratedClassifierCV, we need to extract the base estimator
# CalibratedClassifierCV wraps the model, so SHAP can't directly use it.
# We'll use the underlying base estimators.

base_model = None

# Try to extract the underlying model from CalibratedClassifierCV
if hasattr(model, "calibrated_classifiers_"):
    # CalibratedClassifierCV stores fitted calibrated classifiers
    # Use the first one's base estimator for SHAP
    # But for more accurate SHAP, refit the best uncalibrated model
    log("  Detected CalibratedClassifierCV wrapper.")
    log("  Extracting base estimator for SHAP analysis...")

    # The base estimator from the first fold
    first_cal = model.calibrated_classifiers_[0]
    base_model = first_cal.estimator

    # Better approach: refit the uncalibrated model on full training data
    # using the same best params
    if hasattr(base_model, "get_params"):
        log("  Refitting base model on full training data for SHAP...")
        from sklearn.base import clone
        base_model_fresh = clone(base_model)
        base_model_fresh.fit(X_train_use, train_df["HIGH_COST"].values)
        base_model = base_model_fresh
        log("  Done.")
else:
    base_model = model

# Choose appropriate explainer
model_type = type(base_model).__name__
log(f"  Base model type: {model_type}")

if model_type in ["XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]:
    log("  Using TreeExplainer (exact, fast)")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_test_use)

    # For binary classifiers, TreeExplainer may return a list [class0, class1]
    # or a single array (for XGBoost with binary:logistic)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (high-cost)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

elif model_type == "LogisticRegression":
    log("  Using LinearExplainer")
    explainer = shap.LinearExplainer(base_model, X_train_use)
    shap_values = explainer.shap_values(X_test_use)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]
else:
    log("  Using KernelExplainer (slower, model-agnostic)")
    # Sample background data for efficiency
    bg = shap.sample(X_train_use, 100, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(base_model.predict_proba, bg)
    shap_values = explainer.shap_values(X_test_use[:500])  # limit for speed
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

log(f"  SHAP values shape: {shap_values.shape}")
log(f"  Expected value (base rate in log-odds/margin): {expected_value:.4f}")


# =============================================================================
# 3. GLOBAL FEATURE IMPORTANCE (mean |SHAP|)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 3: GLOBAL FEATURE IMPORTANCE")
log("=" * 72)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

log(f"\nTop {TOP_N_FEATURES} features by mean |SHAP|:")
for i, (feat, val) in enumerate(feat_importance.head(TOP_N_FEATURES).items()):
    log(f"  {i+1:>2}. {feat:<30s}  {val:.4f}")


# =============================================================================
# 4. FEATURE GROUP IMPORTANCE
# =============================================================================
log("\n" + "=" * 72)
log("STEP 4: FEATURE GROUP IMPORTANCE")
log("=" * 72)

# Define feature groups
feature_groups = OrderedDict([
    ("Demographics", [c for c in feature_cols if c.startswith(("AGE", "FEMALE", "RACE_", "MARRIED", "WIDOWED", "DIVORCED", "EDUCYR", "REGION_"))]),
    ("Socioeconomic", [c for c in feature_cols if c.startswith(("POVCAT", "LOW_INCOME"))]),
    ("Insurance", [c for c in feature_cols if c.startswith(("INS_", "UNINSURED"))]),
    ("Health Status", [c for c in feature_cols if c.startswith(("PHYS_", "MENT_"))]),
    ("Chronic Conditions", [c for c in feature_cols if c.startswith(("DX_", "N_CHRONIC", "MULTIMORBID", "CVD_"))]),
    ("Utilization", [c for c in feature_cols if c.startswith(("UTIL_", "ANY_ER", "ANY_INPAT"))]),
    ("Expenditure", [c for c in feature_cols if c.startswith(("EXP_", "ANY_SPENDING"))]),
    ("Interactions", [c for c in feature_cols if c.startswith(("AGE_x_", "ELDERLY_", "PUBLIC_"))]),
])

group_importance = {}
for group, cols in feature_groups.items():
    col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
    if col_indices:
        group_shap = np.abs(shap_values[:, col_indices]).sum(axis=1).mean()
        group_importance[group] = group_shap
        log(f"  {group:<25s}: {group_shap:.4f}  ({len(col_indices)} features)")

# Verify all features are assigned
assigned = sum(len(v) for v in feature_groups.values())
log(f"\n  Features assigned to groups: {assigned}/{len(feature_cols)}")
unassigned = [c for c in feature_cols if not any(c in v for v in feature_groups.values())]
if unassigned:
    log(f"  Unassigned: {unassigned}")


# =============================================================================
# 5. DIRECTION OF EFFECTS (top features)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 5: DIRECTION OF EFFECTS (Top Features)")
log("=" * 72)

top_feats = feat_importance.head(TOP_N_FEATURES).index.tolist()
log(f"\nFor the top {TOP_N_FEATURES} features, positive SHAP = pushes toward HIGH_COST:\n")

for feat in top_feats[:10]:
    idx = feature_cols.index(feat)
    feat_vals = X_test_df[feat].values
    feat_shap = shap_values[:, idx]

    # Correlation between feature value and SHAP value
    corr = np.corrcoef(feat_vals, feat_shap)[0, 1]
    direction = "HIGHER value → HIGHER risk" if corr > 0 else "HIGHER value → LOWER risk"

    # Mean SHAP for high vs low feature values
    median_val = np.median(feat_vals)
    mean_shap_high = feat_shap[feat_vals > median_val].mean()
    mean_shap_low = feat_shap[feat_vals <= median_val].mean()

    log(f"  {feat:<30s}  corr={corr:+.3f}  ({direction})")
    log(f"    {'':>30s}  SHAP when high={mean_shap_high:+.4f}, when low={mean_shap_low:+.4f}")


# =============================================================================
# 6. INDIVIDUAL EXPLANATIONS (example patients)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 6: INDIVIDUAL PATIENT EXPLANATIONS")
log("=" * 72)

# Find a correctly predicted high-cost patient with high confidence
proba_test = base_model.predict_proba(X_test_use)[:, 1]
hc_mask = y_test == 1
nhc_mask = y_test == 0

# High-cost example: true positive with highest predicted probability
hc_indices = np.where(hc_mask)[0]
hc_probas = proba_test[hc_indices]
example_hc_idx = hc_indices[np.argmax(hc_probas)]

# Low-cost example: true negative with lowest predicted probability
nhc_indices = np.where(nhc_mask)[0]
nhc_probas = proba_test[nhc_indices]
example_nhc_idx = nhc_indices[np.argmin(nhc_probas)]

for label, idx in [("HIGH-COST (True Positive)", example_hc_idx),
                   ("LOW-COST (True Negative)", example_nhc_idx)]:
    log(f"\n  Example {label} — Test index {idx}:")
    log(f"    Predicted probability: {proba_test[idx]:.4f}")
    log(f"    Actual next-year expenditure: ${test_df.iloc[idx]['TOTEXP_NEXT']:,.0f}")

    # Top 5 features pushing toward high-cost
    patient_shap = shap_values[idx]
    top_pos = np.argsort(patient_shap)[::-1][:5]
    log(f"    Top drivers TOWARD high-cost:")
    for i in top_pos:
        log(f"      {feature_cols[i]:<30s}  value={X_test_df.iloc[idx][feature_cols[i]]:>10.1f}  "
            f"SHAP={patient_shap[i]:+.4f}")

    # Top 5 features pushing toward low-cost
    top_neg = np.argsort(patient_shap)[:5]
    log(f"    Top drivers TOWARD low-cost:")
    for i in top_neg:
        log(f"      {feature_cols[i]:<30s}  value={X_test_df.iloc[idx][feature_cols[i]]:>10.1f}  "
            f"SHAP={patient_shap[i]:+.4f}")


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 7: GENERATING VISUALIZATIONS")
log("=" * 72)


# ---- FIGURE 1: Bar plot (global importance) ----
fig, ax = plt.subplots(figsize=(10, 8))
top_imp = feat_importance.head(TOP_N_FEATURES).sort_values()
colors = ["#E53935" if v > feat_importance.quantile(0.9) else "#1565C0"
          for v in top_imp.values]
top_imp.plot(kind="barh", ax=ax, color=colors, alpha=0.85)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title(f"Top {TOP_N_FEATURES} Features — Global Importance (mean |SHAP|)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved shap_summary_bar.png")


# ---- FIGURE 2: Beeswarm (SHAP value distribution by feature value) ----
fig, ax = plt.subplots(figsize=(10, 8))
# Use raw feature values for coloring (not scaled)
shap.summary_plot(
    shap_values,
    X_test_df,
    max_display=TOP_N_FEATURES,
    show=False,
    plot_size=None,
)
plt.title(f"SHAP Beeswarm — Feature Effects on High-Cost Prediction",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png"), bbox_inches="tight", dpi=150)
plt.close("all")
log("  Saved shap_summary_beeswarm.png")


# ---- FIGURE 3: Dependence plots for top 6 features ----
top6 = feat_importance.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("SHAP Dependence Plots — Top 6 Features", fontsize=14, fontweight="bold")

for i, feat in enumerate(top6):
    ax = axes[i // 3, i % 3]
    feat_idx = feature_cols.index(feat)

    # Find best interaction feature automatically
    shap_col = shap_values[:, feat_idx]
    feat_val = X_test_df[feat].values

    # Simple scatter
    scatter = ax.scatter(
        feat_val, shap_col,
        c=X_test_df[feat].values,
        cmap="RdBu_r", s=8, alpha=0.5,
        vmin=np.percentile(feat_val, 5),
        vmax=np.percentile(feat_val, 95),
    )
    ax.set_xlabel(feat)
    ax.set_ylabel("SHAP Value")
    ax.set_title(feat, fontsize=10)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_top6.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved shap_dependence_top6.png")


# ---- FIGURE 4: Force plots (saved as bar charts for compatibility) ----
for label, idx, fname in [
    ("High-Cost Patient (True Positive)", example_hc_idx, "shap_force_high_cost.png"),
    ("Low-Cost Patient (True Negative)", example_nhc_idx, "shap_force_low_cost.png"),
]:
    fig, ax = plt.subplots(figsize=(10, 6))
    patient_shap = shap_values[idx]
    sorted_idx = np.argsort(np.abs(patient_shap))[::-1][:15]

    feats = [feature_cols[j] for j in sorted_idx]
    vals = [patient_shap[j] for j in sorted_idx]
    feat_vals = [X_test_df.iloc[idx][feature_cols[j]] for j in sorted_idx]

    colors = ["#E53935" if v > 0 else "#1565C0" for v in vals]
    labels = [f"{f}\n(={fv:.0f})" for f, fv in zip(feats, feat_vals)]

    y_pos = np.arange(len(feats))
    ax.barh(y_pos, vals, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SHAP Value (red=toward high-cost, blue=toward low-cost)")
    ax.axvline(0, color="black", lw=0.5)

    actual_exp = test_df.iloc[idx]["TOTEXP_NEXT"]
    pred_prob = proba_test[idx]
    ax.set_title(f"{label}\nPredicted Prob: {pred_prob:.3f} | Actual Expenditure: ${actual_exp:,.0f}",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"  Saved {fname}")


# ---- FIGURE 5: Feature Group Importance ----
fig, ax = plt.subplots(figsize=(10, 5))
grp_series = pd.Series(group_importance).sort_values()
grp_colors = ["#E53935" if v > grp_series.quantile(0.75) else "#1565C0"
              for v in grp_series.values]
grp_series.plot(kind="barh", ax=ax, color=grp_colors, alpha=0.85)
ax.set_xlabel("Mean Sum of |SHAP| Values")
ax.set_title("Feature Group Importance — Which Domains Drive Predictions?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_group_importance.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved shap_group_importance.png")


# ---- FIGURE 6: SHAP interaction plot (top 2 features) ----
top2 = feat_importance.head(2).index.tolist()
fig, ax = plt.subplots(figsize=(9, 7))
feat1_idx = feature_cols.index(top2[0])
feat2_idx = feature_cols.index(top2[1])

scatter = ax.scatter(
    X_test_df[top2[0]], X_test_df[top2[1]],
    c=shap_values[:, feat1_idx] + shap_values[:, feat2_idx],
    cmap="RdBu_r", s=12, alpha=0.6,
)
plt.colorbar(scatter, ax=ax, label="Combined SHAP Effect")
ax.set_xlabel(top2[0])
ax.set_ylabel(top2[1])
ax.set_title(f"SHAP Interaction: {top2[0]} × {top2[1]}\n(color = combined SHAP effect on high-cost risk)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_interaction_top.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved shap_interaction_top.png")


# =============================================================================
# 8. SAVE SHAP VALUES
# =============================================================================
log("\n" + "=" * 72)
log("STEP 8: SAVING SHAP VALUES")
log("=" * 72)

shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{c}" for c in feature_cols])
shap_df.insert(0, "DUPERSID", test_df["DUPERSID"].values)
shap_df.insert(1, "HIGH_COST", y_test)
shap_df.insert(2, "TOTEXP_NEXT", test_df["TOTEXP_NEXT"].values)
shap_df.insert(3, "PREDICTED_PROB", proba_test)
shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_values.csv"), index=False)
log(f"  Saved shap_values.csv ({shap_df.shape})")

# Summary text
with open(os.path.join(OUTPUT_DIR, "shap_analysis_summary.txt"), "w") as f:
    f.write("\n".join(log_lines))
log("  Saved shap_analysis_summary.txt")

log(f"\n{'='*72}")
log("ALL DONE! SHAP analysis complete.")
log(f"Outputs in {OUTPUT_DIR}/")
log(f"{'='*72}")