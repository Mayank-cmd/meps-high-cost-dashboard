"""
=============================================================================
MEPS High-Cost Patient — Step 2: Feature Engineering
=============================================================================
Team 1 | MISM 6212

Prerequisites:
    pip install pandas numpy scikit-learn

Input:
    ./eda_output/cohort1_2021_2022.csv
    ./eda_output/cohort2_2022_2023.csv

Output:
    ./features_output/train_features.csv      (Cohort 1 — model-ready)
    ./features_output/test_features.csv       (Cohort 2 — model-ready)
    ./features_output/feature_dictionary.txt  (description of every feature)

Usage:
    python step2_feature_engineering.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR = "./eda_output"
OUTPUT_DIR = "./features_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_cohort(filename):
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path)
    print(f"Loaded {filename}: {df.shape}")
    return df


def engineer_features(df, cohort_name=""):
    """
    Transform raw MEPS variables into model-ready features.

    Design principles:
      - No target leakage: all features come from Year t, target from Year t+1
      - Handle MEPS reserved codes (-1, -7, -8, -9, -15) explicitly
      - Create clinically meaningful derived features
      - Log-transform skewed expenditure/utilization variables
      - One-hot encode categorical variables
    """
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING: {cohort_name}")
    print(f"{'='*60}")

    out = pd.DataFrame()
    out["DUPERSID"] = df["DUPERSID"]

    # ------------------------------------------------------------------
    # 1. DEMOGRAPHICS
    # ------------------------------------------------------------------
    print("  [1/8] Demographics...")

    # Age: replace -1 (inapplicable) with median; create age groups
    age = df["AGE"].copy()
    age = age.replace({-1: np.nan})
    age = age.fillna(age.median())
    out["AGE"] = age
    out["AGE_SQ"] = age ** 2  # quadratic term for nonlinear age effects

    # Age buckets (for interpretability and interaction detection)
    out["AGE_0_17"] = (age < 18).astype(int)
    out["AGE_18_34"] = ((age >= 18) & (age < 35)).astype(int)
    out["AGE_35_49"] = ((age >= 35) & (age < 50)).astype(int)
    out["AGE_50_64"] = ((age >= 50) & (age < 65)).astype(int)
    out["AGE_65_PLUS"] = (age >= 65).astype(int)

    # Sex: 1=Male, 2=Female → binary female indicator
    out["FEMALE"] = (df["SEX"] == 2).astype(int)

    # Race/ethnicity: one-hot (reference = White Non-Hispanic)
    out["RACE_HISPANIC"] = (df["RACETHX"] == 1).astype(int)
    out["RACE_BLACK"] = (df["RACETHX"] == 3).astype(int)
    out["RACE_ASIAN"] = (df["RACETHX"] == 4).astype(int)
    out["RACE_OTHER"] = (df["RACETHX"] == 5).astype(int)

    # Marital status: collapse to meaningful groups
    marry = df["MARRY"].copy()
    out["MARRIED"] = (marry == 1).astype(int)
    out["WIDOWED"] = (marry == 2).astype(int)
    out["DIVORCED_SEP"] = (marry.isin([3, 4])).astype(int)
    # Reference = never married / unknown / inapplicable

    # Education: handle -1 (inapplicable, children), -7 (refused), -8 (DK)
    educyr = df["EDUCYR"].copy()
    educyr = educyr.replace({-1: np.nan, -7: np.nan, -8: np.nan})
    educyr = educyr.fillna(educyr.median())
    out["EDUCYR"] = educyr

    # Region: one-hot (reference = Northeast=1)
    region = df["REGION"].replace({-1: np.nan}).fillna(df["REGION"].mode()[0])
    out["REGION_MIDWEST"] = (region == 2).astype(int)
    out["REGION_SOUTH"] = (region == 3).astype(int)
    out["REGION_WEST"] = (region == 4).astype(int)

    # ------------------------------------------------------------------
    # 2. SOCIOECONOMIC
    # ------------------------------------------------------------------
    print("  [2/8] Socioeconomic...")

    # Poverty category: ordinal (1=Poor ... 5=High Income)
    out["POVCAT"] = df["POVCAT"]
    # Also create binary: poor/near-poor indicator
    out["LOW_INCOME"] = (df["POVCAT"].isin([1, 2])).astype(int)

    # ------------------------------------------------------------------
    # 3. INSURANCE
    # ------------------------------------------------------------------
    print("  [3/8] Insurance...")

    # INSCOV: 1=Any private, 2=Public only, 3=Uninsured
    out["INS_PRIVATE"] = (df["INSCOV"] == 1).astype(int)
    out["INS_PUBLIC"] = (df["INSCOV"] == 2).astype(int)
    out["UNINSURED"] = (df["INSCOV"] == 3).astype(int)

    # ------------------------------------------------------------------
    # 4. HEALTH STATUS
    # ------------------------------------------------------------------
    print("  [4/8] Health status...")

    # Perceived physical health: 1=Excellent ... 5=Poor
    # Handle negative codes → treat as missing → impute with median
    for var, out_name in [("RTHLTH", "PHYS_HEALTH"), ("MNHLTH", "MENT_HEALTH")]:
        vals = df[var].copy()
        vals = vals.replace({-1: np.nan, -7: np.nan, -8: np.nan, -9: np.nan})
        vals = vals.fillna(vals.median())
        out[out_name] = vals

    # Binary: fair/poor health indicators
    out["PHYS_FAIR_POOR"] = (out["PHYS_HEALTH"] >= 4).astype(int)
    out["MENT_FAIR_POOR"] = (out["MENT_HEALTH"] >= 4).astype(int)

    # ------------------------------------------------------------------
    # 5. CHRONIC CONDITIONS
    # ------------------------------------------------------------------
    print("  [5/8] Chronic conditions...")

    chronic_cols = ["HIBPDX", "CHDDX", "OHRTDX", "STRKDX",
                    "EMPHDX", "CANCERDX", "DIABDX", "ARTHDX"]
    chronic_out_names = ["DX_HIGHBP", "DX_CHD", "DX_OTHERHD", "DX_STROKE",
                         "DX_EMPHYSEMA", "DX_CANCER", "DX_DIABETES", "DX_ARTHRITIS"]

    for raw_col, out_name in zip(chronic_cols, chronic_out_names):
        vals = df[raw_col].copy()
        # -1 = inapplicable (age < 18): treat as 0 (no diagnosis)
        # -7/-8/-15 = refused/DK/cannot compute: treat as 0 (conservative)
        vals = (vals == 1).astype(int)
        out[out_name] = vals

    # Derived: multimorbidity count and flags
    chronic_feature_cols = chronic_out_names
    out["N_CHRONIC"] = out[chronic_feature_cols].sum(axis=1)
    out["MULTIMORBID_2PLUS"] = (out["N_CHRONIC"] >= 2).astype(int)
    out["MULTIMORBID_3PLUS"] = (out["N_CHRONIC"] >= 3).astype(int)

    # Cardiovascular cluster (CHD + Other HD + Stroke + High BP)
    out["CVD_CLUSTER"] = out[["DX_CHD", "DX_OTHERHD", "DX_STROKE", "DX_HIGHBP"]].sum(axis=1)

    # ------------------------------------------------------------------
    # 6. UTILIZATION (prior year)
    # ------------------------------------------------------------------
    print("  [6/8] Utilization...")

    util_cols = ["OBTOTV", "OPTOTV", "ERTOT", "IPDIS", "RXTOT", "DVTOT"]
    util_out_names = ["UTIL_OFFICE", "UTIL_OUTPAT", "UTIL_ER", "UTIL_INPAT",
                      "UTIL_RX", "UTIL_DENTAL"]

    for raw_col, out_name in zip(util_cols, util_out_names):
        vals = df[raw_col].copy().clip(lower=0)  # floor negatives at 0
        out[out_name] = vals
        # Log-transformed version (log1p handles zeros)
        out[f"{out_name}_LOG"] = np.log1p(vals)

    # Derived: any ER visit, any inpatient stay
    out["ANY_ER"] = (out["UTIL_ER"] > 0).astype(int)
    out["ANY_INPAT"] = (out["UTIL_INPAT"] > 0).astype(int)

    # Total utilization across all types
    out["UTIL_TOTAL"] = out[util_out_names].sum(axis=1)
    out["UTIL_TOTAL_LOG"] = np.log1p(out["UTIL_TOTAL"])

    # ------------------------------------------------------------------
    # 7. EXPENDITURE (prior year — current year as feature)
    # ------------------------------------------------------------------
    print("  [7/8] Expenditure features...")

    exp_cols = ["TOTEXP_CURR", "OBVEXP", "OPTEXP", "ERTEXP", "IPFEXP", "RXEXP"]
    exp_out_names = ["EXP_TOTAL", "EXP_OFFICE", "EXP_OUTPAT", "EXP_ER",
                     "EXP_INPAT", "EXP_RX"]

    for raw_col, out_name in zip(exp_cols, exp_out_names):
        vals = df[raw_col].copy().clip(lower=0)
        out[out_name] = vals
        out[f"{out_name}_LOG"] = np.log1p(vals)

    # Expenditure concentration: share of spending in each category
    total_exp = out["EXP_TOTAL"].replace(0, np.nan)
    for out_name in ["EXP_OFFICE", "EXP_OUTPAT", "EXP_ER", "EXP_INPAT", "EXP_RX"]:
        out[f"{out_name}_SHARE"] = (out[out_name] / total_exp).fillna(0)

    # Binary: any spending at all
    out["ANY_SPENDING"] = (out["EXP_TOTAL"] > 0).astype(int)

    # ------------------------------------------------------------------
    # 8. INTERACTION FEATURES
    # ------------------------------------------------------------------
    print("  [8/8] Interaction features...")

    # Age × chronic condition count (older + sicker = higher risk)
    out["AGE_x_NCHRONIC"] = out["AGE"] * out["N_CHRONIC"]

    # Age × prior expenditure (elderly with high prior spending)
    out["AGE_x_EXP_LOG"] = out["AGE"] * out["EXP_TOTAL_LOG"]

    # Elderly + multimorbid
    out["ELDERLY_MULTIMORBID"] = (out["AGE_65_PLUS"] * out["MULTIMORBID_2PLUS"])

    # Public insurance + poor health
    out["PUBLIC_INS_POORHLTH"] = (out["INS_PUBLIC"] * out["PHYS_FAIR_POOR"])

    # ------------------------------------------------------------------
    # ATTACH TARGET & METADATA
    # ------------------------------------------------------------------
    out["TOTEXP_NEXT"] = df["TOTEXP_NEXT"]     # continuous target
    out["HIGH_COST"] = df["HIGH_COST"]          # binary target
    out["PERWT"] = df["PERWT"]                  # survey weight
    out["VARSTR"] = df["VARSTR"]                # variance stratum
    out["VARPSU"] = df["VARPSU"]                # variance PSU

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    feature_cols = [c for c in out.columns if c not in
                    ["DUPERSID", "TOTEXP_NEXT", "HIGH_COST", "PERWT", "VARSTR", "VARPSU"]]

    print(f"\n  Total features created: {len(feature_cols)}")
    print(f"  Rows: {len(out):,}")
    print(f"  Any nulls remaining: {out[feature_cols].isnull().any().any()}")
    print(f"  Target distribution: {out['HIGH_COST'].value_counts().to_dict()}")

    return out, feature_cols


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # Load cohorts from EDA step
    c1 = load_cohort("cohort1_2021_2022.csv")
    c2 = load_cohort("cohort2_2022_2023.csv")

    # Engineer features
    train_df, feature_names = engineer_features(c1, "Cohort 1 (Train: 2021→2022)")
    test_df, _ = engineer_features(c2, "Cohort 2 (Test: 2022→2023)")

    # Save
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_features.csv"), index=False)

    # ------------------------------------------------------------------
    # Feature dictionary
    # ------------------------------------------------------------------
    feature_dict = {
        # Demographics
        "AGE": "Age in years (end of year, imputed)",
        "AGE_SQ": "Age squared (captures nonlinear age effects)",
        "AGE_0_17": "Indicator: age 0-17",
        "AGE_18_34": "Indicator: age 18-34",
        "AGE_35_49": "Indicator: age 35-49",
        "AGE_50_64": "Indicator: age 50-64",
        "AGE_65_PLUS": "Indicator: age 65+",
        "FEMALE": "Indicator: female (1) vs male (0)",
        "RACE_HISPANIC": "Indicator: Hispanic ethnicity",
        "RACE_BLACK": "Indicator: Black non-Hispanic",
        "RACE_ASIAN": "Indicator: Asian non-Hispanic",
        "RACE_OTHER": "Indicator: Other/multiple race non-Hispanic",
        "MARRIED": "Indicator: currently married",
        "WIDOWED": "Indicator: widowed",
        "DIVORCED_SEP": "Indicator: divorced or separated",
        "EDUCYR": "Years of education (imputed for missing)",
        "REGION_MIDWEST": "Indicator: Midwest census region",
        "REGION_SOUTH": "Indicator: South census region",
        "REGION_WEST": "Indicator: West census region",
        # Socioeconomic
        "POVCAT": "Family income as % of poverty line (1=Poor..5=High)",
        "LOW_INCOME": "Indicator: poor or near-poor (POVCAT 1-2)",
        # Insurance
        "INS_PRIVATE": "Indicator: any private insurance coverage",
        "INS_PUBLIC": "Indicator: public insurance only (Medicaid/Medicare)",
        "UNINSURED": "Indicator: uninsured full year",
        # Health status
        "PHYS_HEALTH": "Perceived physical health (1=Excellent..5=Poor)",
        "MENT_HEALTH": "Perceived mental health (1=Excellent..5=Poor)",
        "PHYS_FAIR_POOR": "Indicator: physical health fair or poor",
        "MENT_FAIR_POOR": "Indicator: mental health fair or poor",
        # Chronic conditions
        "DX_HIGHBP": "Diagnosis: high blood pressure",
        "DX_CHD": "Diagnosis: coronary heart disease",
        "DX_OTHERHD": "Diagnosis: other heart disease",
        "DX_STROKE": "Diagnosis: stroke",
        "DX_EMPHYSEMA": "Diagnosis: emphysema",
        "DX_CANCER": "Diagnosis: cancer",
        "DX_DIABETES": "Diagnosis: diabetes",
        "DX_ARTHRITIS": "Diagnosis: arthritis",
        "N_CHRONIC": "Count of chronic conditions (0-8)",
        "MULTIMORBID_2PLUS": "Indicator: 2+ chronic conditions",
        "MULTIMORBID_3PLUS": "Indicator: 3+ chronic conditions",
        "CVD_CLUSTER": "Count of cardiovascular conditions (CHD+OtherHD+Stroke+HighBP, 0-4)",
        # Utilization
        "UTIL_OFFICE": "Office-based provider visits (count)",
        "UTIL_OUTPAT": "Outpatient department visits (count)",
        "UTIL_ER": "Emergency room visits (count)",
        "UTIL_INPAT": "Inpatient hospital discharges (count)",
        "UTIL_RX": "Prescription medicine fills/refills (count)",
        "UTIL_DENTAL": "Dental care visits (count)",
        "UTIL_OFFICE_LOG": "Log(1 + office visits)",
        "UTIL_OUTPAT_LOG": "Log(1 + outpatient visits)",
        "UTIL_ER_LOG": "Log(1 + ER visits)",
        "UTIL_INPAT_LOG": "Log(1 + inpatient discharges)",
        "UTIL_RX_LOG": "Log(1 + Rx fills)",
        "UTIL_DENTAL_LOG": "Log(1 + dental visits)",
        "ANY_ER": "Indicator: at least 1 ER visit",
        "ANY_INPAT": "Indicator: at least 1 inpatient stay",
        "UTIL_TOTAL": "Total utilization events across all categories",
        "UTIL_TOTAL_LOG": "Log(1 + total utilization)",
        # Expenditure
        "EXP_TOTAL": "Total healthcare expenditure ($, current year)",
        "EXP_OFFICE": "Office-based expenditure ($)",
        "EXP_OUTPAT": "Outpatient expenditure ($)",
        "EXP_ER": "ER expenditure ($)",
        "EXP_INPAT": "Inpatient facility expenditure ($)",
        "EXP_RX": "Prescription medicine expenditure ($)",
        "EXP_TOTAL_LOG": "Log(1 + total expenditure)",
        "EXP_OFFICE_LOG": "Log(1 + office expenditure)",
        "EXP_OUTPAT_LOG": "Log(1 + outpatient expenditure)",
        "EXP_ER_LOG": "Log(1 + ER expenditure)",
        "EXP_INPAT_LOG": "Log(1 + inpatient expenditure)",
        "EXP_RX_LOG": "Log(1 + Rx expenditure)",
        "EXP_OFFICE_SHARE": "Share of total expenditure from office visits",
        "EXP_OUTPAT_SHARE": "Share of total expenditure from outpatient",
        "EXP_ER_SHARE": "Share of total expenditure from ER",
        "EXP_INPAT_SHARE": "Share of total expenditure from inpatient",
        "EXP_RX_SHARE": "Share of total expenditure from Rx",
        "ANY_SPENDING": "Indicator: any healthcare spending > $0",
        # Interactions
        "AGE_x_NCHRONIC": "Age × chronic condition count",
        "AGE_x_EXP_LOG": "Age × log(total expenditure)",
        "ELDERLY_MULTIMORBID": "Indicator: age 65+ AND 2+ chronic conditions",
        "PUBLIC_INS_POORHLTH": "Indicator: public insurance AND fair/poor health",
    }

    dict_path = os.path.join(OUTPUT_DIR, "feature_dictionary.txt")
    with open(dict_path, "w") as f:
        f.write("MEPS High-Cost Patient — Feature Dictionary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total features: {len(feature_names)}\n")
        f.write(f"Train set: {len(train_df):,} rows\n")
        f.write(f"Test set:  {len(test_df):,} rows\n\n")

        current_group = ""
        for feat in feature_names:
            desc = feature_dict.get(feat, "—")
            # Group headers
            if feat == "AGE" and current_group != "DEMO":
                f.write("\n--- DEMOGRAPHICS ---\n")
                current_group = "DEMO"
            elif feat == "POVCAT" and current_group != "SOCIO":
                f.write("\n--- SOCIOECONOMIC ---\n")
                current_group = "SOCIO"
            elif feat == "INS_PRIVATE" and current_group != "INS":
                f.write("\n--- INSURANCE ---\n")
                current_group = "INS"
            elif feat == "PHYS_HEALTH" and current_group != "HLTH":
                f.write("\n--- HEALTH STATUS ---\n")
                current_group = "HLTH"
            elif feat == "DX_HIGHBP" and current_group != "CHRONIC":
                f.write("\n--- CHRONIC CONDITIONS ---\n")
                current_group = "CHRONIC"
            elif feat == "UTIL_OFFICE" and current_group != "UTIL":
                f.write("\n--- UTILIZATION ---\n")
                current_group = "UTIL"
            elif feat == "EXP_TOTAL" and current_group != "EXP":
                f.write("\n--- EXPENDITURE ---\n")
                current_group = "EXP"
            elif feat == "AGE_x_NCHRONIC" and current_group != "INTER":
                f.write("\n--- INTERACTIONS ---\n")
                current_group = "INTER"

            f.write(f"  {feat:<30s}  {desc}\n")

    print(f"\n{'='*60}")
    print(f"ALL DONE! Outputs saved to {OUTPUT_DIR}/")
    print(f"  - train_features.csv  ({len(train_df):,} rows × {len(feature_names)} features)")
    print(f"  - test_features.csv   ({len(test_df):,} rows × {len(feature_names)} features)")
    print(f"  - feature_dictionary.txt")
    print(f"{'='*60}")