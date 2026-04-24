"""
=============================================================================
MEPS High-Cost Patient — Step 5b: SHAP-Based Clustering & Risk Personas
=============================================================================
Team 1 | MISM 6212

This version clusters on SHAP values instead of raw features.
Patients are grouped by *what drives their risk* rather than just
their demographics — producing more meaningful and differentiated personas.

Prerequisites:
    pip install pandas numpy scikit-learn matplotlib seaborn

Input:
    ./features_output/test_features.csv
    ./tuned_output/predictions_test_tuned.csv
    ./shap_output/shap_values.csv

Output:
    ./cluster_output/cluster_assignments.csv
    ./cluster_output/cluster_elbow_silhouette.png
    ./cluster_output/cluster_pca_scatter.png
    ./cluster_output/cluster_heatmap.png
    ./cluster_output/cluster_profiles.png
    ./cluster_output/cluster_shap_drivers.png
    ./cluster_output/cluster_analysis_summary.txt

Usage:
    python step5b_shap_clustering.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import OrderedDict

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
FEATURES_DIR = "./features_output"
TUNED_DIR = "./tuned_output"
SHAP_DIR = "./shap_output"
OUTPUT_DIR = "./cluster_output"
RANDOM_STATE = 42
TOP_K_PCT = 20  # cluster among top 20% highest-risk

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

test_df = pd.read_csv(os.path.join(FEATURES_DIR, "test_features.csv"))
preds_df = pd.read_csv(os.path.join(TUNED_DIR, "predictions_test_tuned.csv"))
shap_df = pd.read_csv(os.path.join(SHAP_DIR, "shap_values.csv"))

log(f"Test set: {test_df.shape}")
log(f"Predictions: {preds_df.shape}")
log(f"SHAP values: {shap_df.shape}")

# Find best model probability column
proba_cols = [c for c in preds_df.columns if c.startswith("proba_")]
best_proba_col = None
for c in proba_cols:
    if "xgboost" in c and "calibrated" in c:
        best_proba_col = c
        break
if best_proba_col is None:
    best_proba_col = proba_cols[-1]
log(f"Using: {best_proba_col}")

preds_df["PRED_PROB"] = preds_df[best_proba_col]


# =============================================================================
# 2. SELECT HIGH-RISK SUBSET
# =============================================================================
log("\n" + "=" * 72)
log(f"STEP 2: SELECTING HIGH-RISK SUBSET (Top {TOP_K_PCT}%)")
log("=" * 72)

threshold = preds_df["PRED_PROB"].quantile(1 - TOP_K_PCT / 100)
high_risk_mask = preds_df["PRED_PROB"] >= threshold
high_risk_idx = preds_df[high_risk_mask].index

log(f"Total test patients: {len(preds_df):,}")
log(f"Risk score threshold: {threshold:.4f}")
log(f"High-risk subset: {len(high_risk_idx):,}")
log(f"True high-cost captured: {preds_df.loc[high_risk_idx, 'HIGH_COST'].sum()}"
    f" / {preds_df['HIGH_COST'].sum()}"
    f" ({preds_df.loc[high_risk_idx, 'HIGH_COST'].sum() / preds_df['HIGH_COST'].sum() * 100:.1f}%)")


# =============================================================================
# 3. PREPARE SHAP-BASED CLUSTERING FEATURES
# =============================================================================
log("\n" + "=" * 72)
log("STEP 3: PREPARING SHAP-BASED CLUSTERING FEATURES")
log("=" * 72)

shap_cols = [c for c in shap_df.columns if c.startswith("SHAP_")]
shap_names = [c.replace("SHAP_", "") for c in shap_cols]

# Extract SHAP values for high-risk subset
shap_high_risk = shap_df.loc[high_risk_idx, shap_cols].copy()
log(f"SHAP matrix for clustering: {shap_high_risk.shape}")

# Feature selection: use top SHAP features by variance within the high-risk group
# High variance in SHAP = this feature differentiates patients within the high-risk pool
shap_variance = shap_high_risk.var().sort_values(ascending=False)

log(f"\nTop 20 SHAP features by variance (most differentiating):")
for i, (col, var) in enumerate(shap_variance.head(20).items()):
    log(f"  {i+1:>2}. {col.replace('SHAP_', ''):<30s}  variance={var:.4f}")

# Select top features for clustering (those that actually differentiate patients)
N_CLUSTER_FEATURES = 15
top_shap_cols = shap_variance.head(N_CLUSTER_FEATURES).index.tolist()
top_shap_names = [c.replace("SHAP_", "") for c in top_shap_cols]

X_cluster_raw = shap_high_risk[top_shap_cols].values
log(f"\nUsing top {N_CLUSTER_FEATURES} SHAP features for clustering")
log(f"Selected features: {top_shap_names}")

# Scale for K-Means
clust_scaler = StandardScaler()
X_cluster = clust_scaler.fit_transform(X_cluster_raw)


# =============================================================================
# 4. OPTIMAL K SELECTION
# =============================================================================
log("\n" + "=" * 72)
log("STEP 4: SELECTING OPTIMAL K (Elbow + Silhouette)")
log("=" * 72)

K_range = range(2, 9)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_cluster, labels)
    silhouettes.append(sil)
    log(f"  K={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}")

best_k = list(K_range)[np.argmax(silhouettes)]
log(f"\nBest K by silhouette: {best_k}")

CHOSEN_K = max(best_k, 3)
if best_k > 5:
    CHOSEN_K = 4
log(f"Chosen K for analysis: {CHOSEN_K}")


# =============================================================================
# 5. FIT FINAL CLUSTERING
# =============================================================================
log("\n" + "=" * 72)
log(f"STEP 5: FITTING K-MEANS (K={CHOSEN_K}) ON SHAP VALUES")
log("=" * 72)

km_final = KMeans(n_clusters=CHOSEN_K, n_init=30, random_state=RANDOM_STATE)
cluster_labels = km_final.fit_predict(X_cluster)
sil_final = silhouette_score(X_cluster, cluster_labels)
log(f"Final silhouette score: {sil_final:.4f}")
log(f"Cluster sizes: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

# Attach labels
high_risk_df = test_df.loc[high_risk_idx].copy()
high_risk_df["CLUSTER"] = cluster_labels
high_risk_df["PRED_PROB"] = preds_df.loc[high_risk_idx, "PRED_PROB"].values
high_risk_df["TOTEXP_NEXT"] = preds_df.loc[high_risk_idx, "TOTEXP_NEXT"].values
high_risk_df["HIGH_COST"] = preds_df.loc[high_risk_idx, "HIGH_COST"].values


# =============================================================================
# 6. PROFILE EACH CLUSTER
# =============================================================================
log("\n" + "=" * 72)
log("STEP 6: CLUSTER PROFILES")
log("=" * 72)

cluster_ids = sorted(high_risk_df["CLUSTER"].unique())

for c in cluster_ids:
    grp = high_risk_df[high_risk_df["CLUSTER"] == c]
    pct = len(grp) / len(high_risk_df) * 100
    log(f"\n{'─'*60}")
    log(f"CLUSTER {c} (n={len(grp)}, {pct:.1f}% of high-risk subset)")
    log(f"{'─'*60}")
    log(f"  Mean Age                    : {grp['AGE'].mean():.1f}")
    log(f"  % Female                    : {grp['FEMALE'].mean()*100:.1f}%")
    log(f"  % Private Insurance         : {grp['INS_PRIVATE'].mean()*100:.1f}%")
    log(f"  % Public Insurance          : {grp['INS_PUBLIC'].mean()*100:.1f}%")
    log(f"  Mean Phys Health (1-5)      : {grp['PHYS_HEALTH'].mean():.1f}")
    log(f"  Mean Mental Health (1-5)    : {grp['MENT_HEALTH'].mean():.1f}")
    log(f"  Mean # Chronic Conditions   : {grp['N_CHRONIC'].mean():.1f}")
    log(f"  % 3+ Chronic                : {grp['MULTIMORBID_3PLUS'].mean()*100:.1f}%")
    log(f"  % Diabetes                  : {grp['DX_DIABETES'].mean()*100:.1f}%")
    log(f"  % Cancer                    : {grp['DX_CANCER'].mean()*100:.1f}%")
    log(f"  Mean Office Visits          : {grp['UTIL_OFFICE'].mean():.1f}")
    log(f"  Mean ER Visits              : {grp['UTIL_ER'].mean():.1f}")
    log(f"  Mean IP Discharges          : {grp['UTIL_INPAT'].mean():.1f}")
    log(f"  Mean Rx Fills               : {grp['UTIL_RX'].mean():.1f}")
    log(f"  Mean Total Exp (Yr t)       : ${grp['EXP_TOTAL'].mean():,.0f}")
    log(f"  Mean Inpatient Exp          : ${grp['EXP_INPAT'].mean():,.0f}")
    log(f"  Mean Rx Exp                 : ${grp['EXP_RX'].mean():,.0f}")
    log(f"  Mean Predicted Prob         : {grp['PRED_PROB'].mean():.3f}")
    log(f"  True High-Cost Rate         : {grp['HIGH_COST'].mean()*100:.1f}%")
    log(f"  Mean Actual Next-Yr Exp     : ${grp['TOTEXP_NEXT'].mean():,.0f}")


# =============================================================================
# 7. PERSONA NAMING (with insurance differentiation)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 7: PERSONA NAMING & INTERVENTION MAPPING")
log("=" * 72)

persona_names = {}
persona_interventions = {}

for c in cluster_ids:
    grp = high_risk_df[high_risk_df["CLUSTER"] == c]
    traits = []

    # Age
    if grp["AGE"].mean() >= 65:
        traits.append("Elderly")
    elif grp["AGE"].mean() >= 50:
        traits.append("Older Adult")
    elif grp["AGE"].mean() < 35:
        traits.append("Younger")
    else:
        traits.append("Middle-Aged")

    # Condition burden
    if grp["N_CHRONIC"].mean() >= 3:
        traits.append("Multimorbid")
    elif grp["N_CHRONIC"].mean() >= 1.5:
        traits.append("Moderate Conditions")
    else:
        traits.append("Low Condition Burden")

    # Insurance (key differentiator)
    if grp["INS_PRIVATE"].mean() > 0.7:
        traits.append("Private Insurance")
    elif grp["INS_PUBLIC"].mean() > 0.7:
        traits.append("Public Insurance")
    else:
        traits.append("Mixed Insurance")

    # Spending pattern
    if grp["EXP_INPAT"].mean() > grp["EXP_RX"].mean() and grp["EXP_INPAT"].mean() > 5000:
        traits.append("Inpatient-Driven")
    elif grp["EXP_RX"].mean() > grp["EXP_INPAT"].mean() and grp["EXP_RX"].mean() > 5000:
        traits.append("Rx-Driven")
    elif grp["EXP_TOTAL"].mean() > 30000:
        traits.append("High Spend")

    # ER utilization
    if grp["UTIL_ER"].mean() > 0.6:
        traits.append("High ER Use")

    persona_name = " / ".join(traits)
    persona_names[c] = persona_name

    # Intervention mapping
    interventions = []
    if grp["N_CHRONIC"].mean() >= 2:
        interventions.append("Care coordination for multimorbidity")
    if grp["UTIL_RX"].mean() > 30:
        interventions.append("Medication therapy management")
    if grp["UTIL_ER"].mean() > 0.5:
        interventions.append("ER diversion / urgent care redirection")
    if grp["UTIL_INPAT"].mean() > 0.3:
        interventions.append("Transitional care / discharge planning")
    if grp["DX_DIABETES"].mean() > 0.3:
        interventions.append("Diabetes management program")
    if grp["PHYS_HEALTH"].mean() >= 3.5:
        interventions.append("Chronic disease self-management support")
    if grp["MENT_HEALTH"].mean() >= 3.5:
        interventions.append("Behavioral health integration")
    if grp["INS_PUBLIC"].mean() > 0.6:
        interventions.append("Benefits navigation / social services")
    if grp["AGE"].mean() >= 65:
        interventions.append("Medicare wellness / preventive screening")
    if not interventions:
        interventions.append("General preventive outreach")

    persona_interventions[c] = interventions

    log(f"\n  Cluster {c}: \"{persona_name}\"")
    log(f"  Suggested interventions:")
    for intv in interventions:
        log(f"    - {intv}")


# =============================================================================
# 8. TOP SHAP DRIVERS PER CLUSTER
# =============================================================================
log("\n" + "=" * 72)
log("STEP 8: TOP SHAP DRIVERS PER CLUSTER")
log("=" * 72)

shap_all_cols = [c for c in shap_df.columns if c.startswith("SHAP_")]
shap_all_names = [c.replace("SHAP_", "") for c in shap_all_cols]
shap_hr = shap_df.loc[high_risk_idx].copy()
shap_hr["CLUSTER"] = cluster_labels

for c in cluster_ids:
    grp_shap = shap_hr[shap_hr["CLUSTER"] == c][shap_all_cols]
    mean_shap = grp_shap.mean()
    mean_shap.index = shap_all_names

    top_pos = mean_shap.sort_values(ascending=False).head(5)
    top_neg = mean_shap.sort_values(ascending=True).head(3)

    log(f"\n  Cluster {c} (\"{persona_names[c]}\"):")
    log(f"    Top drivers TOWARD high-cost:")
    for feat, val in top_pos.items():
        log(f"      {feat:<30s}  mean SHAP = {val:+.4f}")
    log(f"    Top drivers TOWARD low-cost:")
    for feat, val in top_neg.items():
        log(f"      {feat:<30s}  mean SHAP = {val:+.4f}")


# =============================================================================
# 9. VISUALIZATIONS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 9: GENERATING VISUALIZATIONS")
log("=" * 72)

cluster_colors = ["#1565C0", "#E53935", "#43A047", "#FF8F00", "#7B1FA2"]


# ---- Elbow + Silhouette ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cluster Selection Diagnostics (SHAP-Based)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(list(K_range), inertias, "bo-", lw=2)
ax.axvline(CHOSEN_K, color="red", ls="--", alpha=0.7, label=f"Chosen K={CHOSEN_K}")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
ax.legend()

ax = axes[1]
ax.plot(list(K_range), silhouettes, "go-", lw=2)
ax.axvline(CHOSEN_K, color="red", ls="--", alpha=0.7, label=f"Chosen K={CHOSEN_K}")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Analysis")
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cluster_elbow_silhouette.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved cluster_elbow_silhouette.png")


# ---- PCA Scatter ----
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_cluster)

fig, ax = plt.subplots(figsize=(9, 7))
for c in cluster_ids:
    mask = cluster_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=cluster_colors[c], s=15, alpha=0.5,
               label=f"Cluster {c+1}: {persona_names[c][:35]}")

centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c="black", marker="X", s=200, edgecolors="white", lw=2,
           label="Centroids", zorder=5)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Risk Personas — PCA of SHAP-Based Clustering", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cluster_pca_scatter.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved cluster_pca_scatter.png")


# ---- Heatmap ----
profile_feats = [
    "AGE", "FEMALE", "PHYS_HEALTH", "MENT_HEALTH",
    "N_CHRONIC", "DX_DIABETES", "DX_CANCER", "DX_HIGHBP", "DX_ARTHRITIS",
    "UTIL_OFFICE", "UTIL_ER", "UTIL_INPAT", "UTIL_RX",
    "EXP_TOTAL", "EXP_INPAT", "EXP_RX",
    "INS_PUBLIC", "POVCAT",
]
means_by_cluster = high_risk_df.groupby("CLUSTER")[profile_feats].mean()
means_z = (means_by_cluster - means_by_cluster.mean()) / (means_by_cluster.std() + 1e-8)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(means_z.T, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
            ax=ax, linewidths=0.5, cbar_kws={"label": "Z-Score"},
            xticklabels=[f"C{c+1}: {persona_names[c][:25]}" for c in cluster_ids])
ax.set_title("Cluster Feature Profiles (Standardized)", fontsize=13, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cluster_heatmap.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved cluster_heatmap.png")


# ---- Radar Chart ----
radar_feats = ["AGE", "N_CHRONIC", "PHYS_HEALTH", "UTIL_RX",
               "UTIL_ER", "UTIL_INPAT", "EXP_TOTAL", "EXP_INPAT", "EXP_RX"]
radar_labels = ["Age", "# Chronic", "Poor Health", "Rx Fills",
                "ER Visits", "IP Stays", "Total Exp", "IP Exp", "Rx Exp"]

means_raw = high_risk_df.groupby("CLUSTER")[radar_feats].mean()
means_norm = (means_raw - means_raw.min()) / (means_raw.max() - means_raw.min() + 1e-8)

n_vars = len(radar_feats)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)

for c in cluster_ids:
    vals = means_norm.loc[c].values.tolist()
    vals += vals[:1]
    ax.plot(angles, vals, "o-", lw=2, color=cluster_colors[c],
            label=f"C{c+1}: {persona_names[c][:30]}", alpha=0.8)
    ax.fill(angles, vals, color=cluster_colors[c], alpha=0.1)

ax.set_title("Risk Persona Profiles (Normalized)", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cluster_profiles.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved cluster_profiles.png")


# ---- SHAP Drivers per Cluster ----
fig, axes = plt.subplots(1, CHOSEN_K, figsize=(6 * CHOSEN_K, 7))
if CHOSEN_K == 1:
    axes = [axes]
fig.suptitle("Top SHAP Drivers by Cluster", fontsize=14, fontweight="bold")

for c in cluster_ids:
    ax = axes[c]
    grp_shap = shap_hr[shap_hr["CLUSTER"] == c][shap_all_cols]
    mean_shap = grp_shap.mean()
    mean_shap.index = shap_all_names

    top10 = mean_shap.abs().sort_values(ascending=False).head(10)
    top10_vals = mean_shap[top10.index].sort_values()

    colors_bar = [cluster_colors[c] if v > 0 else "#90A4AE" for v in top10_vals.values]
    top10_vals.plot(kind="barh", ax=ax, color=colors_bar, alpha=0.85)
    ax.set_xlabel("Mean SHAP Value")
    ax.set_title(f"C{c+1}: {persona_names[c][:30]}", fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cluster_shap_drivers.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved cluster_shap_drivers.png")


# =============================================================================
# 10. SAVE OUTPUTS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 10: SAVING OUTPUTS")
log("=" * 72)

assign_df = high_risk_df[["DUPERSID", "HIGH_COST", "TOTEXP_NEXT", "PRED_PROB", "CLUSTER"]].copy()
assign_df["PERSONA"] = assign_df["CLUSTER"].map(persona_names)
assign_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_assignments.csv"), index=False)
log(f"  Saved cluster_assignments.csv ({len(assign_df):,} patients)")

with open(os.path.join(OUTPUT_DIR, "cluster_analysis_summary.txt"), "w") as f:
    f.write("\n".join(log_lines))
log("  Saved cluster_analysis_summary.txt")

log(f"\n{'='*72}")
log("ALL DONE! SHAP-based clustering complete.")
log(f"  {CHOSEN_K} risk personas identified")
log(f"  Silhouette score: {sil_final:.4f}")
log(f"  Clustering method: K-Means on top {N_CLUSTER_FEATURES} SHAP features by variance")

# Check for duplicate names
unique_names = set(persona_names.values())
if len(unique_names) < len(persona_names):
    log(f"\n  NOTE: Some personas share similar names. Review profiles for differentiation.")
else:
    log(f"\n  All {len(persona_names)} personas have unique names.")

log(f"\nPersonas:")
for c in cluster_ids:
    log(f"  Cluster {c+1}: {persona_names[c]}")

log(f"\nOutputs in {OUTPUT_DIR}/")
log(f"{'='*72}")