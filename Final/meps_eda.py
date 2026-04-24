"""
=============================================================================
MEPS High-Cost Patient Early Warning — Exploratory Data Analysis
=============================================================================
Team 1 | MISM 6212 | Data Mining / Machine Learning for Business

Prerequisites:
    pip install pandas numpy matplotlib seaborn

Usage:
    1. Place h233.dta, h243.dta, h251.dta in the same directory as this script
       (or update DATA_DIR below)
    2. Run:  python meps_eda.py
    3. Outputs will be saved to ./eda_output/

Outputs:
    - cohort1_2021_2022.csv   (training cohort: features from 2021, target from 2022)
    - cohort2_2022_2023.csv   (test cohort: features from 2022, target from 2023)
    - fig1_spending_distribution.png
    - fig2_risk_factor_profiles.png
    - fig3_correlations.png
    - fig4_cohort_comparison.png
    - eda_summary.txt          (full text summary of all findings)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — Update these paths if your files are elsewhere
# =============================================================================
DATA_DIR = "/Users/mayankbhardwaj/Documents/Semester_2/MISM-6212/GroupProject/CLD/"                      # directory containing .dta files
OUTPUT_DIR = "./eda_output"         # where to save all outputs
HIGH_COST_PERCENTILE = 0.95         # top 5% = high-cost threshold

# Plot styling
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")
COLORS = {"hc": "#E53935", "nhc": "#43A047", "c1": "#1565C0", "c2": "#FF8F00"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
log_lines = []  # collect text output for summary file

def log(text=""):
    """Print and store text for the summary file."""
    print(text)
    log_lines.append(text)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
log("=" * 72)
log("STEP 1: LOADING MEPS DATA")
log("=" * 72)

df21 = pd.read_stata(os.path.join(DATA_DIR, "h233.dta"), convert_categoricals=False)
log(f"2021 (h233): {df21.shape[0]:,} rows × {df21.shape[1]:,} cols")

df22 = pd.read_stata(os.path.join(DATA_DIR, "h243.dta"), convert_categoricals=False)
log(f"2022 (h243): {df22.shape[0]:,} rows × {df22.shape[1]:,} cols")

df23 = pd.read_stata(os.path.join(DATA_DIR, "h251.dta"), convert_categoricals=False)
log(f"2023 (h251): {df23.shape[0]:,} rows × {df23.shape[1]:,} cols")

for name, df in [("2021", df21), ("2022", df22), ("2023", df23)]:
    log(f"  {name} panels: {df['PANEL'].value_counts().sort_index().to_dict()}")


# =============================================================================
# 2. BUILD PAIRED COHORTS (Year t features → Year t+1 target)
# =============================================================================
log("\n" + "=" * 72)
log("STEP 2: COHORT CONSTRUCTION")
log("=" * 72)


def extract_features(df, year_suffix, cohort_ids):
    """
    Extract and standardize features from a single MEPS year.
    
    Year-specific variable names (e.g., TOTEXP21, INSCOV22) are mapped
    to generic names (TOTEXP_CURR, INSCOV) so both cohorts share a
    consistent schema.
    """
    sub = df[df["DUPERSID"].isin(cohort_ids)].copy()
    ys = year_suffix  # '21', '22', or '23'

    rename_map = {
        # --- Identifiers ---
        "DUPERSID": "DUPERSID",
        "PANEL": "PANEL",
        # --- Demographics ---
        f"AGE{ys}X": "AGE",
        "SEX": "SEX",
        "RACETHX": "RACETHX",
        f"MARRY{ys}X": "MARRY",
        "EDUCYR": "EDUCYR",
        f"REGION{ys}": "REGION",
        # --- Socioeconomic ---
        f"POVCAT{ys}": "POVCAT",
        # --- Insurance ---
        f"INSCOV{ys}": "INSCOV",
        # --- Self-reported health (end-of-year round) ---
        "RTHLTH53": "RTHLTH",
        "MNHLTH53": "MNHLTH",
        # --- Chronic condition diagnoses (adults only; -1 = inapplicable/under 18) ---
        "HIBPDX": "HIBPDX",
        "CHDDX": "CHDDX",
        "OHRTDX": "OHRTDX",
        "STRKDX": "STRKDX",
        "EMPHDX": "EMPHDX",
        "CANCERDX": "CANCERDX",
        "DIABDX_M18": "DIABDX",
        "ARTHDX": "ARTHDX",
        # --- Utilization counts ---
        f"OBTOTV{ys}": "OBTOTV",    # office-based visits
        f"OPTOTV{ys}": "OPTOTV",    # outpatient dept visits
        f"ERTOT{ys}": "ERTOT",      # ER visits
        f"IPDIS{ys}": "IPDIS",      # inpatient discharges
        f"RXTOT{ys}": "RXTOT",      # Rx fills
        f"DVTOT{ys}": "DVTOT",      # dental visits
        # --- Current-year expenditure (as feature, NOT target) ---
        f"TOTEXP{ys}": "TOTEXP_CURR",
        f"OBVEXP{ys}": "OBVEXP",     # office-based expenditure
        f"OPTEXP{ys}": "OPTEXP",     # outpatient expenditure
        f"ERTEXP{ys}": "ERTEXP",     # ER expenditure
        f"IPFEXP{ys}": "IPFEXP",     # inpatient facility expenditure
        f"RXEXP{ys}": "RXEXP",       # Rx expenditure
        # --- Survey design variables (for weighted analysis) ---
        f"PERWT{ys}F": "PERWT",
        "VARSTR": "VARSTR",
        "VARPSU": "VARPSU",
    }

    # Only keep columns that actually exist in the data
    existing = {}
    for src, tgt in rename_map.items():
        matches = [c for c in sub.columns if c.upper() == src.upper()]
        if matches:
            existing[matches[0]] = tgt

    return sub[list(existing.keys())].rename(columns=existing)


# Find individuals present in consecutive years
ids_21, ids_22, ids_23 = set(df21["DUPERSID"]), set(df22["DUPERSID"]), set(df23["DUPERSID"])
overlap_21_22 = ids_21 & ids_22
overlap_22_23 = ids_22 & ids_23

log(f"Individuals in both 2021 & 2022: {len(overlap_21_22):,}")
log(f"Individuals in both 2022 & 2023: {len(overlap_22_23):,}")

# Cohort 1: features from 2021, target from 2022
feat_21 = extract_features(df21, "21", overlap_21_22)
target_22 = df22[df22["DUPERSID"].isin(overlap_21_22)][["DUPERSID", "TOTEXP22"]].copy()
target_22.rename(columns={"TOTEXP22": "TOTEXP_NEXT"}, inplace=True)
cohort1 = feat_21.merge(target_22, on="DUPERSID", how="inner")

# Cohort 2: features from 2022, target from 2023
feat_22 = extract_features(df22, "22", overlap_22_23)
target_23 = df23[df23["DUPERSID"].isin(overlap_22_23)][["DUPERSID", "TOTEXP23"]].copy()
target_23.rename(columns={"TOTEXP23": "TOTEXP_NEXT"}, inplace=True)
cohort2 = feat_22.merge(target_23, on="DUPERSID", how="inner")

# Create high-cost binary target
for name, c in [("Cohort 1 (2021→2022)", cohort1), ("Cohort 2 (2022→2023)", cohort2)]:
    threshold = c["TOTEXP_NEXT"].quantile(HIGH_COST_PERCENTILE)
    c["HIGH_COST"] = (c["TOTEXP_NEXT"] >= threshold).astype(int)
    log(f"\n{name}:")
    log(f"  N = {len(c):,}")
    log(f"  High-cost threshold (top {(1-HIGH_COST_PERCENTILE)*100:.0f}%): ${threshold:,.0f}")
    log(f"  High-cost count: {c['HIGH_COST'].sum()} ({c['HIGH_COST'].mean()*100:.1f}%)")
    log(f"  Mean next-year expenditure: ${c['TOTEXP_NEXT'].mean():,.0f}")

# Save cohort CSVs
cohort1.to_csv(os.path.join(OUTPUT_DIR, "cohort1_2021_2022.csv"), index=False)
cohort2.to_csv(os.path.join(OUTPUT_DIR, "cohort2_2022_2023.csv"), index=False)
log(f"\nCohort CSVs saved to {OUTPUT_DIR}/")


# =============================================================================
# 3. SPENDING DISTRIBUTION & CONCENTRATION
# =============================================================================
log("\n" + "=" * 72)
log("STEP 3: SPENDING DISTRIBUTION & CONCENTRATION")
log("=" * 72)

for name, c in [("Cohort 1 (2021→2022)", cohort1), ("Cohort 2 (2022→2023)", cohort2)]:
    exp = c["TOTEXP_NEXT"]
    log(f"\n{name} — Next-Year Total Expenditure:")
    log(f"  Mean:   ${exp.mean():>12,.0f}")
    log(f"  Median: ${exp.median():>12,.0f}")
    log(f"  Std:    ${exp.std():>12,.0f}")
    log(f"  Min:    ${exp.min():>12,.0f}")
    log(f"  Max:    ${exp.max():>12,.0f}")
    log(f"  Zero expenditure: {(exp == 0).sum():,} ({(exp == 0).mean()*100:.1f}%)")

    sorted_exp = exp.sort_values(ascending=False)
    total = sorted_exp.sum()
    for pct in [1, 5, 10, 20, 50]:
        n = max(1, int(len(sorted_exp) * pct / 100))
        share = sorted_exp.iloc[:n].sum() / total * 100
        log(f"  Top {pct:>2}% accounts for {share:>5.1f}% of total spending")


# =============================================================================
# 4. DEMOGRAPHIC & CLINICAL PROFILES
# =============================================================================
log("\n" + "=" * 72)
log("STEP 4: HIGH-COST vs NON-HIGH-COST PROFILES (Cohort 1)")
log("=" * 72)

hc = cohort1[cohort1["HIGH_COST"] == 1]
nhc = cohort1[cohort1["HIGH_COST"] == 0]

# Age
log("\n--- AGE ---")
log(f"  High-cost mean:     {hc['AGE'][hc['AGE'] >= 0].mean():.1f}")
log(f"  Non-high-cost mean: {nhc['AGE'][nhc['AGE'] >= 0].mean():.1f}")

age_bins = [0, 18, 35, 50, 65, 120]
age_labels = ["0-17", "18-34", "35-49", "50-64", "65+"]
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    valid = grp[grp["AGE"] >= 0]
    dist = pd.cut(valid["AGE"], bins=age_bins, labels=age_labels).value_counts(normalize=True).sort_index()
    log(f"  {grp_name}: {(dist * 100).round(1).to_dict()}")

# Sex
log("\n--- SEX (1=Male, 2=Female) ---")
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    dist = grp["SEX"].value_counts(normalize=True).sort_index()
    log(f"  {grp_name}: Male={dist.get(1,0)*100:.1f}%, Female={dist.get(2,0)*100:.1f}%")

# Race/ethnicity
race_map = {1: "Hispanic", 2: "White Non-Hisp", 3: "Black Non-Hisp", 4: "Asian Non-Hisp", 5: "Other/Multi"}
log("\n--- RACE/ETHNICITY ---")
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    dist = grp["RACETHX"].map(race_map).value_counts(normalize=True)
    log(f"  {grp_name}: {(dist * 100).round(1).to_dict()}")

# Insurance
ins_map = {1: "Any private", 2: "Public only", 3: "Uninsured"}
log("\n--- INSURANCE COVERAGE ---")
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    dist = grp["INSCOV"].map(ins_map).value_counts(normalize=True)
    log(f"  {grp_name}: {(dist * 100).round(1).to_dict()}")

# Perceived health status
hlth_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
log("\n--- PERCEIVED PHYSICAL HEALTH ---")
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    valid = grp[grp["RTHLTH"] > 0]
    dist = valid["RTHLTH"].map(hlth_map).value_counts(normalize=True)
    log(f"  {grp_name}: {dict(sorted({k: round(v*100,1) for k,v in dist.items()}.items()))}")

log("\n--- PERCEIVED MENTAL HEALTH ---")
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    valid = grp[grp["MNHLTH"] > 0]
    dist = valid["MNHLTH"].map(hlth_map).value_counts(normalize=True)
    log(f"  {grp_name}: {dict(sorted({k: round(v*100,1) for k,v in dist.items()}.items()))}")

# Chronic conditions
chronic_cols = ["HIBPDX", "CHDDX", "OHRTDX", "STRKDX", "EMPHDX", "CANCERDX", "DIABDX", "ARTHDX"]
chronic_names = ["High BP", "Coronary HD", "Other HD", "Stroke", "Emphysema", "Cancer", "Diabetes", "Arthritis"]
log("\n--- CHRONIC CONDITION PREVALENCE (% diagnosed) ---")
log(f"  {'Condition':>18s}  {'High-Cost':>10s}  {'Non-HC':>10s}  {'Ratio':>6s}")
for col, cname in zip(chronic_cols, chronic_names):
    hc_pct = (hc[col] == 1).mean() * 100
    nhc_pct = (nhc[col] == 1).mean() * 100
    ratio = hc_pct / nhc_pct if nhc_pct > 0 else float("inf")
    log(f"  {cname:>18s}  {hc_pct:>9.1f}%  {nhc_pct:>9.1f}%  {ratio:>5.1f}x")

# Multimorbidity
log("\n--- MULTIMORBIDITY (adults only) ---")
for grp_name, grp in [("High-cost", hc), ("Non-HC", nhc)]:
    adults = grp[grp["AGE"] >= 18]
    cond_count = (adults[chronic_cols] == 1).sum(axis=1)
    log(f"  {grp_name}: mean={cond_count.mean():.2f}, "
        f"0 conditions={(cond_count == 0).mean()*100:.1f}%, "
        f"3+ conditions={(cond_count >= 3).mean()*100:.1f}%")

# Utilization
util_cols = ["OBTOTV", "OPTOTV", "ERTOT", "IPDIS", "RXTOT", "DVTOT"]
util_names = ["Office visits", "Outpatient visits", "ER visits", "IP discharges", "Rx fills", "Dental visits"]
log("\n--- PRIOR-YEAR UTILIZATION ---")
log(f"  {'Measure':>20s}  {'HC Mean':>10s}  {'NHC Mean':>10s}  {'HC Med':>8s}  {'NHC Med':>8s}")
for col, uname in zip(util_cols, util_names):
    hc_valid = hc[hc[col] >= 0][col]
    nhc_valid = nhc[nhc[col] >= 0][col]
    log(f"  {uname:>20s}  {hc_valid.mean():>10.1f}  {nhc_valid.mean():>10.1f}  "
        f"{hc_valid.median():>8.0f}  {nhc_valid.median():>8.0f}")

# Current-year expenditure
log("\n--- CURRENT-YEAR EXPENDITURE (as feature) ---")
log(f"  High-cost:     mean=${hc['TOTEXP_CURR'].mean():>10,.0f}  median=${hc['TOTEXP_CURR'].median():>10,.0f}")
log(f"  Non-high-cost: mean=${nhc['TOTEXP_CURR'].mean():>10,.0f}  median=${nhc['TOTEXP_CURR'].median():>10,.0f}")


# =============================================================================
# 5. MISSINGNESS & DATA QUALITY
# =============================================================================
log("\n" + "=" * 72)
log("STEP 5: DATA QUALITY & MEPS RESERVED CODES")
log("=" * 72)

feature_cols = [
    "AGE", "SEX", "RACETHX", "MARRY", "EDUCYR", "REGION", "POVCAT", "INSCOV",
    "RTHLTH", "MNHLTH", "HIBPDX", "CHDDX", "OHRTDX", "STRKDX", "EMPHDX",
    "CANCERDX", "DIABDX", "ARTHDX", "OBTOTV", "OPTOTV", "ERTOT", "IPDIS",
    "RXTOT", "DVTOT", "TOTEXP_CURR", "OBVEXP", "OPTEXP", "ERTEXP", "IPFEXP", "RXEXP",
]

for name, c in [("Cohort 1", cohort1), ("Cohort 2", cohort2)]:
    log(f"\n{name} — Null values: {c[feature_cols].isnull().sum().sum()}")
    log(f"{name} — Negative (reserved) codes:")
    for col in feature_cols:
        neg = c[c[col] < 0][col].value_counts()
        if len(neg) > 0:
            log(f"  {col:>12s}: {neg.to_dict()}")

log("""
MEPS Reserved Code Reference:
  -1  = Inapplicable (e.g., chronic condition Qs not asked for age < 18)
  -7  = Refused
  -8  = Don't know
  -9  = Not ascertained
  -15 = Cannot be computed

Recommended handling:
  • Chronic conditions (-1 for minors): recode to 0 or create 'Not Asked' indicator
  • -7/-8 (Refused/Don't Know): use missing indicator given small counts
  • EDUCYR (-1 for children): recode based on age or use missing flag
""")


# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
log("=" * 72)
log("STEP 6: GENERATING VISUALIZATIONS")
log("=" * 72)


# ---- FIGURE 1: Spending Distribution & Concentration ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Spending Distribution & Concentration (Cohort 1: 2021→2022)",
             fontsize=14, fontweight="bold", y=0.98)

# 1a: Log histogram
ax = axes[0, 0]
exp_pos = cohort1["TOTEXP_NEXT"][cohort1["TOTEXP_NEXT"] > 0]
ax.hist(np.log10(exp_pos), bins=50, color="#2196F3", alpha=0.8, edgecolor="white")
threshold = cohort1["TOTEXP_NEXT"].quantile(HIGH_COST_PERCENTILE)
ax.axvline(np.log10(threshold), color="red", ls="--", lw=2,
           label=f"Top 5% (${threshold:,.0f})")
ax.set_xlabel("Log₁₀(Total Expenditure)")
ax.set_ylabel("Count")
ax.set_title("Next-Year Expenditure Distribution (log scale)")
ax.legend()

# 1b: Lorenz curve
ax = axes[0, 1]
sorted_exp = np.sort(cohort1["TOTEXP_NEXT"].values)
cumul = np.cumsum(sorted_exp) / sorted_exp.sum()
pop = np.arange(1, len(sorted_exp) + 1) / len(sorted_exp)
ax.plot(pop, cumul, color="#2196F3", lw=2, label="Actual")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect equality")
ax.fill_between(pop, cumul, pop, alpha=0.15, color="#2196F3")
for pct in [0.80, 0.90, 0.95, 0.99]:
    idx = int(pct * len(sorted_exp))
    ax.plot(pct, cumul[idx - 1], "ro", ms=6)
    ax.annotate(f"Top {(1-pct)*100:.0f}%→{(1-cumul[idx-1])*100:.0f}% spend",
                xy=(pct, cumul[idx - 1]), fontsize=7,
                xytext=(pct - 0.15, cumul[idx - 1] - 0.08))
ax.set_xlabel("Population Percentile")
ax.set_ylabel("Cumulative Share of Spending")
ax.set_title("Healthcare Spending Concentration (Lorenz Curve)")
ax.legend(loc="upper left")

# 1c: Expenditure by category
ax = axes[1, 0]
exp_cats = ["OBVEXP", "OPTEXP", "ERTEXP", "IPFEXP", "RXEXP"]
exp_cat_names = ["Office-Based", "Outpatient", "ER", "Inpatient", "Rx"]
hc_means = [hc[c].mean() for c in exp_cats]
nhc_means = [nhc[c].mean() for c in exp_cats]
x = np.arange(len(exp_cat_names))
ax.bar(x - 0.18, hc_means, 0.35, label="High-Cost", color=COLORS["hc"], alpha=0.85)
ax.bar(x + 0.18, nhc_means, 0.35, label="Non-High-Cost", color=COLORS["nhc"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(exp_cat_names)
ax.set_ylabel("Mean Expenditure ($)")
ax.set_title("Current-Year Spending by Category")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# 1d: Box plot
ax = axes[1, 1]
data_box = [
    nhc["TOTEXP_NEXT"].clip(upper=nhc["TOTEXP_NEXT"].quantile(0.99)),
    hc["TOTEXP_NEXT"].clip(upper=hc["TOTEXP_NEXT"].quantile(0.99)),
]
bp = ax.boxplot(data_box, labels=["Non-High-Cost", "High-Cost"],
                patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor(COLORS["nhc"])
bp["boxes"][1].set_facecolor(COLORS["hc"])
for b in bp["boxes"]:
    b.set_alpha(0.7)
ax.set_ylabel("Next-Year Total Expenditure ($)")
ax.set_title("Next-Year Expenditure by Group (99th pctile clipped)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_spending_distribution.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved fig1_spending_distribution.png")


# ---- FIGURE 2: Risk Factor Profiles ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Risk Factor Profiles: High-Cost vs Non-High-Cost (Cohort 1)",
             fontsize=14, fontweight="bold", y=0.98)

# 2a: Age distribution
ax = axes[0, 0]
for grp_name, grp, color in [("Non-HC", nhc, COLORS["nhc"]), ("High-Cost", hc, COLORS["hc"])]:
    valid = grp[grp["AGE"] >= 0]
    cuts = pd.cut(valid["AGE"], bins=age_bins, labels=age_labels)
    dist = cuts.value_counts(normalize=True).sort_index()
    offset = -0.2 if grp_name == "Non-HC" else 0.2
    ax.bar(np.arange(len(age_labels)) + offset, dist.values * 100, 0.35,
           label=grp_name, color=color, alpha=0.8)
ax.set_xticks(np.arange(len(age_labels)))
ax.set_xticklabels(age_labels)
ax.set_ylabel("Percentage (%)")
ax.set_title("Age Distribution")
ax.legend()

# 2b: Insurance
ax = axes[0, 1]
ins_cats = ["Private", "Public Only", "Uninsured"]
for grp_name, grp, color in [("Non-HC", nhc, COLORS["nhc"]), ("High-Cost", hc, COLORS["hc"])]:
    dist = grp["INSCOV"].map(ins_map).value_counts(normalize=True)
    vals = [dist.get(c, 0) * 100 for c in ins_cats]
    offset = -0.2 if grp_name == "Non-HC" else 0.2
    ax.bar(np.arange(len(ins_cats)) + offset, vals, 0.35,
           label=grp_name, color=color, alpha=0.8)
ax.set_xticks(np.arange(len(ins_cats)))
ax.set_xticklabels(ins_cats)
ax.set_ylabel("Percentage (%)")
ax.set_title("Insurance Coverage")
ax.legend()

# 2c: Perceived health
ax = axes[0, 2]
hlth_labels = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
for grp_name, grp, color in [("Non-HC", nhc, COLORS["nhc"]), ("High-Cost", hc, COLORS["hc"])]:
    valid = grp[grp["RTHLTH"] > 0]
    dist = valid["RTHLTH"].map(hlth_map).value_counts(normalize=True)
    vals = [dist.get(h, 0) * 100 for h in hlth_labels]
    offset = -0.2 if grp_name == "Non-HC" else 0.2
    ax.bar(np.arange(len(hlth_labels)) + offset, vals, 0.35,
           label=grp_name, color=color, alpha=0.8)
ax.set_xticks(np.arange(len(hlth_labels)))
ax.set_xticklabels(hlth_labels, rotation=30, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_title("Perceived Physical Health")
ax.legend()

# 2d: Chronic conditions
ax = axes[1, 0]
hc_prev = [(hc[c] == 1).mean() * 100 for c in chronic_cols]
nhc_prev = [(nhc[c] == 1).mean() * 100 for c in chronic_cols]
y = np.arange(len(chronic_names))
ax.barh(y - 0.2, nhc_prev, 0.35, label="Non-HC", color=COLORS["nhc"], alpha=0.8)
ax.barh(y + 0.2, hc_prev, 0.35, label="High-Cost", color=COLORS["hc"], alpha=0.8)
ax.set_yticks(y)
ax.set_yticklabels(chronic_names)
ax.set_xlabel("Prevalence (%)")
ax.set_title("Chronic Condition Prevalence")
ax.legend()

# 2e: Utilization
ax = axes[1, 1]
util_plot_cols = ["OBTOTV", "ERTOT", "IPDIS", "RXTOT"]
util_plot_names = ["Office\nVisits", "ER\nVisits", "IP\nDischarges", "Rx\nFills"]
hc_u = [hc[hc[c] >= 0][c].mean() for c in util_plot_cols]
nhc_u = [nhc[nhc[c] >= 0][c].mean() for c in util_plot_cols]
x = np.arange(len(util_plot_names))
ax.bar(x - 0.2, nhc_u, 0.35, label="Non-HC", color=COLORS["nhc"], alpha=0.8)
ax.bar(x + 0.2, hc_u, 0.35, label="High-Cost", color=COLORS["hc"], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(util_plot_names)
ax.set_ylabel("Mean Count")
ax.set_title("Prior-Year Utilization")
ax.legend()

# 2f: Current vs next-year scatter
ax = axes[1, 2]
sample = cohort1.sample(min(3000, len(cohort1)), random_state=42)
s_nhc = sample[sample["HIGH_COST"] == 0]
s_hc = sample[sample["HIGH_COST"] == 1]
ax.scatter(np.log10(s_nhc["TOTEXP_CURR"].clip(lower=1)),
           np.log10(s_nhc["TOTEXP_NEXT"].clip(lower=1)),
           alpha=0.3, s=10, color=COLORS["nhc"], label="Non-HC")
ax.scatter(np.log10(s_hc["TOTEXP_CURR"].clip(lower=1)),
           np.log10(s_hc["TOTEXP_NEXT"].clip(lower=1)),
           alpha=0.6, s=20, color=COLORS["hc"], label="High-Cost", marker="x")
ax.set_xlabel("Log₁₀(Current-Year Expenditure)")
ax.set_ylabel("Log₁₀(Next-Year Expenditure)")
ax.set_title("Current vs Next-Year Spending")
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_risk_factor_profiles.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved fig2_risk_factor_profiles.png")


# ---- FIGURE 3: Correlations ----
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Correlations (Cohort 1: 2021→2022)",
             fontsize=14, fontweight="bold", y=0.98)

corr_cols = [
    "AGE", "SEX", "RACETHX", "INSCOV", "RTHLTH", "MNHLTH", "POVCAT",
    "HIBPDX", "CHDDX", "OHRTDX", "STRKDX", "EMPHDX", "CANCERDX", "DIABDX", "ARTHDX",
    "OBTOTV", "ERTOT", "IPDIS", "RXTOT", "TOTEXP_CURR", "HIGH_COST",
]
corr_df = cohort1[corr_cols].copy()
for col in corr_df.columns:
    corr_df.loc[corr_df[col] < 0, col] = np.nan

# 3a: Correlation bar chart with target
ax = axes[0]
corr_target = corr_df.corr()["HIGH_COST"].drop("HIGH_COST").sort_values()
bar_colors = [COLORS["hc"] if v > 0 else "#2196F3" for v in corr_target.values]
corr_target.plot(kind="barh", ax=ax, color=bar_colors, alpha=0.8)
ax.set_xlabel("Pearson Correlation with HIGH_COST")
ax.set_title("Feature Correlation with High-Cost Status")
ax.axvline(0, color="black", lw=0.5)

# 3b: Heatmap
ax = axes[1]
key_feats = ["AGE", "RTHLTH", "MNHLTH", "HIBPDX", "DIABDX", "CANCERDX", "ARTHDX",
             "OBTOTV", "ERTOT", "IPDIS", "RXTOT", "TOTEXP_CURR", "HIGH_COST"]
corr_matrix = corr_df[key_feats].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, vmin=-0.5, vmax=0.5, square=True,
            annot_kws={"size": 7})
ax.set_title("Correlation Heatmap (Key Features)")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_correlations.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved fig3_correlations.png")


# ---- FIGURE 4: Cohort Comparison ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cohort Comparison: 2021→2022 vs 2022→2023",
             fontsize=14, fontweight="bold", y=0.98)

# 4a: Expenditure distributions
ax = axes[0]
for name, c, color in [("2021→2022", cohort1, COLORS["c1"]),
                        ("2022→2023", cohort2, COLORS["c2"])]:
    exp_pos = c["TOTEXP_NEXT"][c["TOTEXP_NEXT"] > 0]
    ax.hist(np.log10(exp_pos), bins=50, alpha=0.5, color=color, label=name, density=True)
ax.set_xlabel("Log₁₀(Next-Year Expenditure)")
ax.set_ylabel("Density")
ax.set_title("Next-Year Expenditure Distribution")
ax.legend()

# 4b: High-cost rate by age group
ax = axes[1]
for name, c, color in [("2021→2022", cohort1, COLORS["c1"]),
                        ("2022→2023", cohort2, COLORS["c2"])]:
    valid = c[c["AGE"] >= 0].copy()
    valid["AGE_GRP"] = pd.cut(valid["AGE"], bins=age_bins, labels=age_labels)
    rates = valid.groupby("AGE_GRP", observed=True)["HIGH_COST"].mean() * 100
    offset = -0.2 if "2021" in name else 0.2
    ax.bar(np.arange(len(age_labels)) + offset, rates.values, 0.35,
           color=color, alpha=0.8, label=name)
ax.set_xticks(np.arange(len(age_labels)))
ax.set_xticklabels(age_labels)
ax.set_ylabel("High-Cost Rate (%)")
ax.set_title("High-Cost Rate by Age Group")
ax.legend()

# 4c: Summary table
ax = axes[2]
ax.axis("off")
rows = []
for name, c in [("2021→2022", cohort1), ("2022→2023", cohort2)]:
    rows.append([
        name,
        f"{len(c):,}",
        f"${c['TOTEXP_NEXT'].mean():,.0f}",
        f"${c['TOTEXP_NEXT'].median():,.0f}",
        f"${c['TOTEXP_NEXT'].quantile(0.95):,.0f}",
        f"{c['HIGH_COST'].mean()*100:.1f}%",
        f"{(c['TOTEXP_NEXT'] == 0).mean()*100:.1f}%",
    ])
tbl = ax.table(
    cellText=rows,
    colLabels=["Cohort", "N", "Mean", "Median", "95th Pct", "HC Rate", "Zero%"],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
ax.set_title("Summary Comparison", pad=20)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig4_cohort_comparison.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved fig4_cohort_comparison.png")


# =============================================================================
# 7. SAVE TEXT SUMMARY
# =============================================================================
summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(log_lines))

log(f"\n{'=' * 72}")
log(f"ALL DONE! Outputs saved to {OUTPUT_DIR}/")
log(f"  - cohort1_2021_2022.csv")
log(f"  - cohort2_2022_2023.csv")
log(f"  - fig1_spending_distribution.png")
log(f"  - fig2_risk_factor_profiles.png")
log(f"  - fig3_correlations.png")
log(f"  - fig4_cohort_comparison.png")
log(f"  - eda_summary.txt")
log(f"{'=' * 72}")