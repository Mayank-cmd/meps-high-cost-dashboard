# 🏥 High-Cost Patient Early Warning System

**Predicting & Targeting the Top 5% of Healthcare Spenders**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://meps-high-cost-dashboard.streamlit.app)

> Built by Team 1 — Meijing Cheng, Xuewen Huang, Steven Spero, Mayank Bhardwaj  
> MISM 6212: Data Mining / Machine Learning for Business | Northeastern University | Spring 2026

---

## The Problem

Healthcare spending is extremely concentrated. The **top 5% of patients account for 47% of all spending**, while the bottom 50% account for less than 3%. Current systems identify high-cost patients only *after* costs accumulate — missing the prevention window entirely.

This project builds an end-to-end analytics system that predicts which patients will become high-cost **next year**, explains why they're at risk, groups them into actionable segments, and quantifies the financial return of targeted intervention.

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | XGBoost (tuned + calibrated) |
| ROC-AUC | 0.863 |
| PR-AUC | 0.396 (8× above baseline) |
| Recall @ Top 10% | 59.3% of true high-cost captured |
| Lift @ Top 1% | 14× better than random |
| Net Savings (100K plan) | $17.9M at 10% capacity |
| Model Advantage vs Random | $31.9M |

## Pipeline

```
meps_eda.py                  → Exploratory data analysis & cohort construction
step2_feature_engineering.py → 78 features across 8 groups
step3_modeling.py            → Baseline models (LR, RF, XGBoost)
step3b_tuning.py             → Hyperparameter tuning & Platt calibration
step4_shap.py                → SHAP explainability analysis
step5b_shap_clustering.py    → SHAP-based risk persona clustering
step6_roi_simulation.py      → ROI / cost-benefit simulation
dashboard.py                 → Interactive Streamlit dashboard
```

## Data

[MEPS](https://meps.ahrq.gov/) (Medical Expenditure Panel Survey) Full-Year Consolidated files for 2021–2023, published by the Agency for Healthcare Research and Quality. Patients appearing in consecutive years are linked via DUPERSID to create two temporal cohorts:

- **Training:** 12,046 patients (features from 2021, target from 2022)
- **Testing:** 8,284 patients (features from 2022, target from 2023)

The target variable is binary: top 5% of next-year total expenditures.

## Approach

### Feature Engineering
78 features organized into demographics, socioeconomic, insurance, health status, chronic conditions (8 diagnoses + multimorbidity), utilization (6 types + logs), expenditure (total + 5 categories + shares + logs), and interaction terms (age × spending, age × conditions).

### Modeling
Three models trained with 80-iteration RandomizedSearchCV optimizing PR-AUC, then calibrated via Platt scaling. XGBoost outperformed after tuning with lower learning rate (0.019), shallower trees (depth 3), and stronger regularization.

### Explainability
SHAP TreeExplainer decomposes every prediction. Prior-year expenditure is the #1 predictor (mean |SHAP| = 0.489), followed by perceived physical health and age × expenditure interaction. Expenditure features collectively contribute 4× more than any other group.

### Risk Personas
K-Means clustering on SHAP values (not raw features) identifies four personas among the top 20% highest-risk patients:

| Persona | N | HC Rate | Key Trait |
|---------|---|---------|-----------|
| C1: Stable Elderly | 487 | 8.8% | Low spend, age-driven risk |
| C2: Acute Episode | 312 | 16.7% | $13K inpatient, episodic |
| C3: Chronic Rx-Driven | 491 | 35.0% | $22K Rx/yr, 47 fills |
| C4: Declining Elder | 367 | 13.1% | Worst health rating (4.3/5) |

### ROI Simulation
Scaled to a 100,000-member health plan ($2,000/member, 15% cost reduction). Cluster 3 generates **+$2.38M** net savings and is the only ROI-positive persona — the strategic recommendation is to prioritize this segment for medication therapy management and care coordination.

## Dashboard

**[meps-high-cost-dashboard.streamlit.app](https://meps-high-cost-dashboard.streamlit.app)**

Five interactive pages:
1. **Overview & EDA** — spending concentration, Lorenz curve, patient profiles
2. **Model Performance** — ROC/PR curves, Recall@K targeting efficiency
3. **What Drives Risk** — SHAP feature importance, individual patient explanations
4. **Risk Personas** — cluster profiles, radar chart, intervention mapping
5. **ROI Simulator** — adjustable sliders for intervention cost, reduction rate, capacity, inflation

## Run Locally

```bash
# Clone
git clone https://github.com/Mayank-cmd/meps-high-cost-dashboard.git
cd meps-high-cost-dashboard

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

To re-run the full pipeline, download MEPS .dta files from [AHRQ](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp) (HC-233, HC-243, HC-251) and run the scripts in order.

## Tech Stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · SHAP · Plotly · Streamlit

## Limitations

- Associations, not causal effects — predictions do not establish intervention efficacy
- MEPS is survey data — performance on actual plan claims requires re-validation
- ROI assumptions are industry benchmarks, not measured from a clinical trial
- No regression-to-the-mean adjustment applied
- Inflation between train/test years not CPI-adjusted at the feature level

---

*Built as part of MISM 6212 at Northeastern University, Spring 2026.*
