"""
=============================================================================
MEPS High-Cost Patient Early Warning — Interactive Dashboard
=============================================================================
Team 1 | MISM 6212 | Data Mining / Machine Learning for Business

Prerequisites:
    pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn

Input (place these folders alongside this file):
    ./eda_output/          (from step 1)
    ./features_output/     (from step 2)
    ./tuned_output/        (from step 3b)
    ./shap_output/         (from step 4)
    ./cluster_output/      (from step 5)

Usage:
    streamlit run dashboard.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG & CUSTOM STYLING
# =============================================================================
st.set_page_config(
    page_title="High-Cost Patient Early Warning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — editorial healthcare aesthetic
# Warm off-white background, deep navy headers, terracotta accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600;8..60,700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        background-color: #FAFAF7;
    }

    /* Main content area */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    /* Headers */
    h1 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #1a2332 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    h2 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #1a2332 !important;
        font-weight: 600 !important;
        font-size: 1.6rem !important;
        letter-spacing: -0.3px !important;
        border-bottom: 2px solid #c4653a;
        padding-bottom: 0.4rem;
        margin-top: 2rem !important;
    }
    h3 {
        font-family: 'DM Sans', sans-serif !important;
        color: #3d5a80 !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        text-transform: uppercase;
        letter-spacing: 1.2px !important;
    }

    /* Body text */
    p, li, .stMarkdown {
        font-family: 'DM Sans', sans-serif !important;
        color: #2c3e50 !important;
        line-height: 1.65 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f0ebe4;
        border-right: 1px solid #d4cdc4;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1a2332 !important;
    }
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stRadio label p {
        color: #1a2332 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
        border-color: #c4653a !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8e0d5;
        border-left: 4px solid #c4653a;
        padding: 12px 16px;
        border-radius: 4px;
    }
    [data-testid="stMetric"] label {
        font-family: 'DM Sans', sans-serif !important;
        color: #6b7b8d !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #1a2332 !important;
        font-weight: 700 !important;
    }

    /* Divider styling */
    hr {
        border: none;
        border-top: 1px solid #d4cdc4;
        margin: 2rem 0;
    }

    /* Tables */
    .stDataFrame {
        border: 1px solid #e8e0d5;
        border-radius: 4px;
    }

    /* Slider */
    .stSlider label {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Callout boxes */
    .callout-box {
        background: white;
        border-left: 4px solid #c4653a;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        font-family: 'DM Sans', sans-serif;
        color: #2c3e50;
    }
    .callout-box strong {
        color: #1a2332;
    }

    .callout-navy {
        background: #1a2332;
        border-left: 4px solid #c4653a;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #e8e0d5 !important;
    }
    .callout-navy p, .callout-navy strong, .callout-navy span {
        color: #e8e0d5 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e8e0d5;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
        padding: 8px 24px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #c4653a !important;
    }

    /* Top header bar — hide completely */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    /* Reclaim the space */
    .block-container {
        padding-top: 1rem !important;
    }
    /* Hide deploy button */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    button[data-testid="stSidebarCollapseButton"] { display: none; }

    /* Expander header — lighter background */
    .streamlit-expanderHeader {
        background-color: #eee8e0 !important;
        border-radius: 4px;
    }
    details[data-testid="stExpander"] summary {
        background-color: #eee8e0 !important;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# COLOR PALETTE
# =============================================================================
COLORS = {
    "navy": "#1a2332",
    "terracotta": "#c4653a",
    "slate": "#3d5a80",
    "sage": "#5a7a62",
    "warm_gray": "#6b7b8d",
    "cream": "#FAFAF7",
    "sand": "#e8e0d5",
    "white": "#ffffff",
    "light_blue": "#dce8f0",
    "error": "#b5473a",
}
PLOT_COLORS = ["#c4653a", "#3d5a80", "#5a7a62", "#d4956a", "#7ba0c4", "#8fb896"]

# Plotly layout template
PLOT_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color="#1a2332", size=14),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAF7",
    margin=dict(l=60, r=30, t=50, b=60),
)

def fix_axes(fig):
    """Force dark, readable axis labels and titles on any Plotly figure."""
    fig.update_xaxes(
        title_font=dict(color="#1a2332", size=14, family="DM Sans, sans-serif"),
        tickfont=dict(color="#1a2332", size=12),
        gridcolor="#bbb",
    )
    fig.update_yaxes(
        title_font=dict(color="#1a2332", size=14, family="DM Sans, sans-serif"),
        tickfont=dict(color="#1a2332", size=12),
        gridcolor="#bbb",
    )
    return fig


# =============================================================================
# DATA LOADING (cached)
# =============================================================================
@st.cache_data
def load_data():
    data = {}

    # EDA cohorts
    try:
        data["cohort1"] = pd.read_csv("./eda_output/cohort1_2021_2022.csv")
        data["cohort2"] = pd.read_csv("./eda_output/cohort2_2022_2023.csv")
    except:
        data["cohort1"] = None
        data["cohort2"] = None

    # Features
    try:
        data["train"] = pd.read_csv("./features_output/train_features.csv")
        data["test"] = pd.read_csv("./features_output/test_features.csv")
    except:
        data["train"] = None
        data["test"] = None

    # Predictions
    try:
        data["preds"] = pd.read_csv("./tuned_output/predictions_test_tuned.csv")
    except:
        data["preds"] = None

    # SHAP
    try:
        data["shap"] = pd.read_csv("./shap_output/shap_values.csv")
    except:
        data["shap"] = None

    # Clusters
    try:
        data["clusters"] = pd.read_csv("./cluster_output/cluster_assignments.csv")
    except:
        data["clusters"] = None

    # Model
    try:
        with open("./tuned_output/best_tuned_model.pkl", "rb") as f:
            data["model_pkg"] = pickle.load(f)
    except:
        data["model_pkg"] = None

    return data


data = load_data()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("### 🏥")
    st.markdown("# Early Warning System")
    st.markdown("*High-Cost Patient Prediction*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview & EDA", "Model Performance", "What Drives Risk", "Risk Personas", "ROI Simulator"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7b8d;'>Team 1 · MISM 6212<br>"
        "Meijing · Xuewen · Steven · Mayank<br>"
        "Data: MEPS 2021–2023</small>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1: OVERVIEW & EDA
# =============================================================================
if page == "Overview & EDA":

    st.markdown("# The Cost Concentration Problem")
    st.markdown(
        "A small fraction of patients generate nearly half of all healthcare spending. "
        "This dashboard presents an early-warning system that identifies those patients "
        "*before* they become high-cost — enabling targeted, proactive intervention."
    )

    # Key metrics row
    if data["cohort1"] is not None:
        c1 = data["cohort1"]
        total_spend = c1["TOTEXP_NEXT"].sum()
        hc = c1[c1["HIGH_COST"] == 1]
        hc_spend = hc["TOTEXP_NEXT"].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patients Studied", f"{len(c1) + len(data['cohort2']):,}")
        col2.metric("High-Cost Threshold", f"${c1['TOTEXP_NEXT'].quantile(0.95):,.0f}")
        col3.metric("Top 5% Share of Spending", f"{hc_spend/total_spend*100:.0f}%")
        col4.metric("Mean High-Cost Expenditure", f"${hc['TOTEXP_NEXT'].mean():,.0f}")

    st.markdown("---")

    ## Spending distribution
    st.markdown("## Where the Money Goes")

    if data["cohort1"] is not None:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            # Lorenz curve
            sorted_exp = np.sort(c1["TOTEXP_NEXT"].values)
            cumul = np.cumsum(sorted_exp) / sorted_exp.sum()
            pop = np.arange(1, len(sorted_exp) + 1) / len(sorted_exp)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pop * 100, y=cumul * 100,
                mode="lines", name="Actual",
                line=dict(color=COLORS["terracotta"], width=3),
                fill="tonexty", fillcolor="rgba(196,101,58,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 100], y=[0, 100],
                mode="lines", name="Perfect equality",
                line=dict(color=COLORS["warm_gray"], width=1, dash="dash"),
            ))

            # Annotate key points
            for pct in [0.90, 0.95, 0.99]:
                idx = int(pct * len(sorted_exp))
                fig.add_annotation(
                    x=pct * 100, y=cumul[idx - 1] * 100,
                    text=f"Top {(1-pct)*100:.0f}% → {(1-cumul[idx-1])*100:.0f}% of spend",
                    showarrow=True, arrowhead=2, arrowsize=0.8,
                    font=dict(size=10, color=COLORS["navy"]),
                    ax=-60, ay=-30,
                )

            fig.update_layout(
                **PLOT_LAYOUT,
                title="Spending Concentration (Lorenz Curve)",
                xaxis_title="Population Percentile",
                yaxis_title="Cumulative Share of Spending (%)",
                showlegend=False,
                height=420,
            )
            fix_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Concentration Facts")
            st.markdown("""
            <div class="callout-box">
                <strong>The top 5%</strong> of patients account for <strong>47%</strong> of all spending.<br><br>
                <strong>The top 1%</strong> alone drives <strong>20%</strong> of total costs.<br><br>
                <strong>The bottom 50%</strong> of patients account for less than <strong>3%</strong>.<br><br>
                <strong>15.5%</strong> of patients incur zero healthcare costs in a given year.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### What This Means")
            st.markdown(
                "If a health plan could identify even *half* of the future top 5% "
                "and intervene to reduce their costs by 15%, the savings would be substantial. "
                "That's what this model attempts to do."
            )

    st.markdown("---")

    ## Profile comparison
    st.markdown("## Who Becomes High-Cost?")

    if data["cohort1"] is not None:
        hc = c1[c1["HIGH_COST"] == 1]
        nhc = c1[c1["HIGH_COST"] == 0]

        profile_data = {
            "Characteristic": [
                "Average Age", "Female (%)", "3+ Chronic Conditions",
                "Fair/Poor Health", "Avg Office Visits", "Avg Rx Fills",
                "Avg Prior-Year Spending", "Public Insurance Only"
            ],
            "High-Cost (Top 5%)": [
                f"{hc['AGE'][hc['AGE']>=0].mean():.0f}",
                f"{(hc['SEX']==2).mean()*100:.0f}%",
                f"{(hc[['HIBPDX','CHDDX','OHRTDX','STRKDX','EMPHDX','CANCERDX','DIABDX','ARTHDX']]==1).sum(axis=1).ge(3).mean()*100:.0f}%",
                f"{hc['RTHLTH'][hc['RTHLTH']>0].ge(4).mean()*100:.0f}%",
                f"{hc['OBTOTV'][hc['OBTOTV']>=0].mean():.0f}",
                f"{hc['RXTOT'][hc['RXTOT']>=0].mean():.0f}",
                f"${hc['TOTEXP_CURR'].mean():,.0f}",
                f"{(hc['INSCOV']==2).mean()*100:.0f}%",
            ],
            "Everyone Else": [
                f"{nhc['AGE'][nhc['AGE']>=0].mean():.0f}",
                f"{(nhc['SEX']==2).mean()*100:.0f}%",
                f"{(nhc[['HIBPDX','CHDDX','OHRTDX','STRKDX','EMPHDX','CANCERDX','DIABDX','ARTHDX']]==1).sum(axis=1).ge(3).mean()*100:.0f}%",
                f"{nhc['RTHLTH'][nhc['RTHLTH']>0].ge(4).mean()*100:.0f}%",
                f"{nhc['OBTOTV'][nhc['OBTOTV']>=0].mean():.0f}",
                f"{nhc['RXTOT'][nhc['RXTOT']>=0].mean():.0f}",
                f"${nhc['TOTEXP_CURR'].mean():,.0f}",
                f"{(nhc['INSCOV']==2).mean()*100:.0f}%",
            ],
        }
        st.dataframe(
            pd.DataFrame(profile_data).set_index("Characteristic"),
            use_container_width=True,
        )


# =============================================================================
# PAGE 2: MODEL PERFORMANCE
# =============================================================================
elif page == "Model Performance":

    st.markdown("# Model Performance")
    st.markdown(
        "Three models were trained on 2021→2022 data and evaluated on a completely "
        "held-out 2022→2023 cohort — a genuine forward-looking test."
    )

    # Metrics summary
    st.markdown("## Evaluation Metrics")

    model_results = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost (best)"],
        "ROC-AUC": [0.861, 0.865, 0.863],
        "PR-AUC": [0.386, 0.390, 0.396],
        "Brier Score": [0.038, 0.038, 0.038],
        "Recall @5%": ["39.8%", "39.5%", "41.2%"],
        "Recall @10%": ["56.9%", "60.5%", "59.3%"],
        "Recall @20%": ["76.1%", "76.6%", "75.9%"],
    }
    st.dataframe(pd.DataFrame(model_results).set_index("Model"), use_container_width=True)

    st.markdown("""
    <div class="callout-box">
        <strong>Recall@K</strong> answers the key operational question: 
        "If we target the top K% riskiest patients, what percentage of the truly 
        high-cost patients do we catch?" Random targeting at 10% would only catch ~10%.
        Our model catches <strong>59%</strong> — a 6× improvement.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if data["preds"] is not None:
        preds = data["preds"]
        y_test = preds["HIGH_COST"].values

        # Find probability columns
        proba_cols = {
            "Logistic Regression": [c for c in preds.columns if "logistic" in c and "calibrated" in c],
            "Random Forest": [c for c in preds.columns if "random_forest" in c and "calibrated" in c],
            "XGBoost": [c for c in preds.columns if "xgboost" in c and "calibrated" in c],
        }
        # Fallback to any matching column
        for name in proba_cols:
            if not proba_cols[name]:
                key = name.lower().replace(" ", "_").split("(")[0].strip("_")
                proba_cols[name] = [c for c in preds.columns if key in c]

        col_left, col_right = st.columns(2)

        ## ROC Curves
        with col_left:
            st.markdown("## ROC Curves")
            fig = go.Figure()
            model_colors = {"Logistic Regression": PLOT_COLORS[1], "Random Forest": PLOT_COLORS[2], "XGBoost": PLOT_COLORS[0]}

            for name, cols in proba_cols.items():
                if cols:
                    y_proba = preds[cols[-1]].values
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                             line=dict(color=model_colors[name], width=2.5)))

            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(color="#c0c0c0", dash="dash", width=1)))
            fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              legend=dict(x=0.5, y=0.05, font=dict(size=11, color="#1a2332")))
            fix_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

        ## Recall@K Curves
        with col_right:
            st.markdown("## Targeting Efficiency")
            fig = go.Figure()
            k_pcts = list(range(1, 31))

            for name, cols in proba_cols.items():
                if cols:
                    y_proba = preds[cols[-1]].values
                    recalls = []
                    for k in k_pcts:
                        n = max(1, int(len(y_test) * k / 100))
                        top_k = np.argsort(y_proba)[::-1][:n]
                        recalls.append(y_test[top_k].sum() / y_test.sum() * 100)
                    fig.add_trace(go.Scatter(x=k_pcts, y=recalls, mode="lines+markers",
                                             name=name, line=dict(color=model_colors[name], width=2.5),
                                             marker=dict(size=4)))

            fig.add_trace(go.Scatter(x=k_pcts, y=k_pcts, mode="lines", name="Random",
                                     line=dict(color="#c0c0c0", dash="dash", width=1)))
            fig.update_layout(**PLOT_LAYOUT, height=400,
                              xaxis_title="Top K% Targeted",
                              yaxis_title="% of True High-Cost Captured",
                              legend=dict(x=0.5, y=0.05, font=dict(size=11, color="#1a2332")))
            fix_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    ## Key observations
    st.markdown("## Key Observations")
    st.markdown("""
    <div class="callout-box">
        <strong>XGBoost went from worst to best after tuning.</strong> 
        With default parameters it underperformed Logistic Regression. 
        A lower learning rate, shallower trees, and stronger regularization fixed the overfitting.
    </div>
    <div class="callout-box">
        <strong>All models are well-calibrated.</strong> 
        After Platt scaling, Brier scores dropped to 0.038 — meaning predicted probabilities 
        are trustworthy for threshold-based decisions.
    </div>
    <div class="callout-box">
        <strong>Temporal validation confirms generalizability.</strong> 
        Cross-validation scores and held-out test scores are very close, indicating the model 
        generalizes across time periods without overfitting.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: WHAT DRIVES RISK (SHAP)
# =============================================================================
elif page == "What Drives Risk":

    st.markdown("# What Drives High-Cost Risk?")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) decomposes every prediction into "
        "individual feature contributions, showing exactly *why* the model flags someone as high-risk."
    )

    # Check if SHAP data exists
    shap_loaded = data["shap"] is not None
    if not shap_loaded:
        st.warning("SHAP data not found. Make sure `./shap_output/shap_values.csv` exists and the dashboard is run from the project root directory.")
        st.info(f"Looking in: `{os.path.abspath('./shap_output/shap_values.csv')}`")
        st.stop()

    shap_df = data["shap"]
    shap_cols = [c for c in shap_df.columns if c.startswith("SHAP_")]
    shap_names = [c.replace("SHAP_", "") for c in shap_cols]

    ## Global importance
    st.markdown("## Global Feature Importance")
    st.markdown("Which factors matter most across all patients, ranked by average impact on predictions.")

    mean_abs = shap_df[shap_cols].abs().mean().values
    imp_df = pd.DataFrame({"Feature": shap_names, "Mean |SHAP|": mean_abs})
    imp_df = imp_df.sort_values("Mean |SHAP|", ascending=True).tail(20)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=imp_df["Feature"], x=imp_df["Mean |SHAP|"],
        orientation="h",
        marker_color=[COLORS["terracotta"] if v > imp_df["Mean |SHAP|"].quantile(0.8) else COLORS["slate"]
                      for v in imp_df["Mean |SHAP|"].values],
    ))
    fig.update_layout(**PLOT_LAYOUT, height=550, title="Top 20 Features by Mean |SHAP Value|",
                      xaxis_title="Mean |SHAP Value|", yaxis_title="", showlegend=False)
    fix_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    ## Feature group importance
    st.markdown("---")
    st.markdown("## Importance by Feature Group")
    st.markdown("Which *domains* of information contribute most to predictions?")

    groups = {
        "Expenditure": [i for i, n in enumerate(shap_names) if n.startswith(("EXP_", "ANY_SPENDING"))],
        "Utilization": [i for i, n in enumerate(shap_names) if n.startswith(("UTIL_", "ANY_ER", "ANY_INPAT"))],
        "Health Status": [i for i, n in enumerate(shap_names) if n.startswith(("PHYS_", "MENT_"))],
        "Interactions": [i for i, n in enumerate(shap_names) if n.startswith(("AGE_x_", "ELDERLY_", "PUBLIC_"))],
        "Chronic Conditions": [i for i, n in enumerate(shap_names) if n.startswith(("DX_", "N_CHRONIC", "MULTIMORBID", "CVD_"))],
        "Demographics": [i for i, n in enumerate(shap_names) if n.startswith(("AGE", "FEMALE", "RACE_", "MARRIED", "WIDOWED", "DIVORCED", "EDUCYR", "REGION_")) and not n.startswith("AGE_x_")],
        "Insurance": [i for i, n in enumerate(shap_names) if n.startswith(("INS_", "UNINSURED"))],
        "Socioeconomic": [i for i, n in enumerate(shap_names) if n.startswith(("POVCAT", "LOW_INCOME"))],
    }

    grp_imp = {}
    shap_vals = shap_df[shap_cols].values
    for grp, indices in groups.items():
        if indices:
            grp_imp[grp] = np.abs(shap_vals[:, indices]).sum(axis=1).mean()

    grp_df = pd.DataFrame({"Group": list(grp_imp.keys()), "Importance": list(grp_imp.values())})
    grp_df = grp_df.sort_values("Importance", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=grp_df["Group"], x=grp_df["Importance"], orientation="h",
        marker_color=[COLORS["terracotta"] if v > grp_df["Importance"].median() else COLORS["slate"]
                      for v in grp_df["Importance"].values],
    ))
    fig.update_layout(**PLOT_LAYOUT, height=350,
                      xaxis_title="Mean Sum of |SHAP Values|", showlegend=False)
    fix_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="callout-box">
        <strong>Prior-year expenditure dominates</strong> — it contributes roughly 4× more 
        predictive power than any other feature group. This makes clinical sense: high costs 
        tend to persist. Utilization patterns and self-reported health status are the next 
        most informative domains.
    </div>
    """, unsafe_allow_html=True)

    ## Individual explanation
    st.markdown("---")
    st.markdown("## Individual Patient Explanations")
    st.markdown("Select a patient type to see what drove the model's prediction.")

    example_type = st.radio("", ["High-Cost Patient (correctly identified)", "Low-Cost Patient (correctly ruled out)"],
                            horizontal=True, label_visibility="collapsed")

    y_test = shap_df["HIGH_COST"].values
    probas = shap_df["PREDICTED_PROB"].values

    if example_type == "High-Cost Patient (correctly identified)":
        hc_idx = np.where(y_test == 1)[0]
        chosen = hc_idx[np.argmax(probas[hc_idx])]
    else:
        nhc_idx = np.where(y_test == 0)[0]
        chosen = nhc_idx[np.argmin(probas[nhc_idx])]

    patient_shap = shap_vals[chosen]
    top_idx = np.argsort(np.abs(patient_shap))[::-1][:12]

    feats = [shap_names[i] for i in top_idx]
    vals = [patient_shap[i] for i in top_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feats[::-1], x=vals[::-1], orientation="h",
        marker_color=[COLORS["terracotta"] if v > 0 else COLORS["slate"] for v in vals[::-1]],
    ))
    fig.update_layout(
        **PLOT_LAYOUT, height=420,
        title=f"Predicted Probability: {probas[chosen]:.1%} | Actual Expenditure: ${shap_df['TOTEXP_NEXT'].iloc[chosen]:,.0f}",
        xaxis_title="SHAP Value (red = toward high-cost, blue = toward low-cost)",
        showlegend=False,
    )
    fig.add_vline(x=0, line_color="#888", line_width=0.5)
    fix_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    ## Direction of effects summary
    st.markdown("---")
    st.markdown("## Direction of Effects")
    st.markdown("For the top predictors, does a *higher* value push toward or away from high-cost?")

    direction_data = []
    for feat_name in imp_df.sort_values("Mean |SHAP|", ascending=False).head(10)["Feature"].values:
        if feat_name in shap_names:
            shap_col_vals = shap_df[f"SHAP_{feat_name}"].values
            # Use SHAP values self-correlation: positive mean SHAP = higher values push toward risk
            mean_shap = shap_col_vals.mean()
            direction = "Higher → Higher Risk" if mean_shap > 0 else "Higher → Lower Risk"
            direction_data.append({
                "Feature": feat_name,
                "Direction": direction,
                "Avg SHAP": f"{mean_shap:+.4f}",
                "Importance": f"{np.abs(shap_col_vals).mean():.4f}",
            })

    if direction_data:
        st.dataframe(pd.DataFrame(direction_data).set_index("Feature"), use_container_width=True)


# =============================================================================
# PAGE 4: RISK PERSONAS
# =============================================================================
elif page == "Risk Personas":

    st.markdown("# Risk Personas")
    st.markdown(
        "Among the top 20% highest-risk patients, clustering reveals distinct subgroups "
        "with different clinical profiles — each suggesting a different intervention strategy."
    )

    if data["clusters"] is not None and data["test"] is not None:
        clusters = data["clusters"]
        test_df = data["test"]

        # Merge cluster labels with test features
        merged = test_df.merge(clusters[["DUPERSID", "CLUSTER", "PERSONA"]], on="DUPERSID", how="inner")

        n_clusters = merged["CLUSTER"].nunique()
        cluster_ids = sorted(merged["CLUSTER"].unique())

        ## Overview metrics
        st.markdown("## At a Glance")
        cols = st.columns(n_clusters)
        for i, c in enumerate(cluster_ids):
            grp = merged[merged["CLUSTER"] == c]
            persona = grp["PERSONA"].iloc[0]
            with cols[i]:
                st.markdown(f"### Cluster {int(c) + 1}")
                st.markdown(f"**{persona}**")
                st.metric("Patients", f"{len(grp):,}")
                st.metric("Avg Age", f"{grp['AGE'].mean():.0f}")
                st.metric("Avg # Chronic", f"{grp['N_CHRONIC'].mean():.1f}")

        st.markdown("---")

        ## Radar chart
        st.markdown("## Profile Comparison")

        radar_feats = ["AGE", "N_CHRONIC", "PHYS_HEALTH", "UTIL_RX",
                       "UTIL_ER", "UTIL_INPAT", "EXP_TOTAL", "EXP_INPAT", "EXP_RX"]
        radar_labels = ["Age", "# Chronic", "Poor Health", "Rx Fills",
                        "ER Visits", "IP Stays", "Total Exp", "IP Exp", "Rx Exp"]

        means = merged.groupby("CLUSTER")[radar_feats].mean()
        means_norm = (means - means.min()) / (means.max() - means.min() + 1e-8)

        fig = go.Figure()
        for i, c in enumerate(cluster_ids):
            vals = means_norm.loc[c].values.tolist()
            vals += vals[:1]
            persona = merged[merged["CLUSTER"] == c]["PERSONA"].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=radar_labels + [radar_labels[0]],
                fill="toself", name=f"Cluster {int(c)+1}: {persona[:30]}",
                line=dict(color=PLOT_COLORS[i], width=2),
                opacity=0.8,
            ))
        fig.update_layout(
            polar=dict(bgcolor="#FAFAF7", radialaxis=dict(visible=True, range=[0, 1], gridcolor="#bbb")),
            **PLOT_LAYOUT, height=520, showlegend=True,
            legend=dict(x=0, y=-0.15, orientation="h", font=dict(size=12, color="#1a2332")),
        )
        fix_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        ## Detailed profiles
        st.markdown("## Detailed Profiles")

        for c in cluster_ids:
            grp = merged[merged["CLUSTER"] == c]
            persona = grp["PERSONA"].iloc[0]

            with st.expander(f"**Cluster {int(c)+1}: {persona}** — {len(grp):,} patients", expanded=bool(c == cluster_ids[0])):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Demographics**")
                    st.markdown(f"- Avg age: {grp['AGE'].mean():.0f}")
                    st.markdown(f"- Female: {grp['FEMALE'].mean()*100:.0f}%")
                    st.markdown(f"- Private insurance: {grp['INS_PRIVATE'].mean()*100:.0f}%")
                    st.markdown(f"- Public insurance: {grp['INS_PUBLIC'].mean()*100:.0f}%")
                with c2:
                    st.markdown("**Clinical Profile**")
                    st.markdown(f"- Chronic conditions: {grp['N_CHRONIC'].mean():.1f}")
                    st.markdown(f"- 3+ chronic: {grp['MULTIMORBID_3PLUS'].mean()*100:.0f}%")
                    st.markdown(f"- Diabetes: {grp['DX_DIABETES'].mean()*100:.0f}%")
                    st.markdown(f"- Physical health (1–5): {grp['PHYS_HEALTH'].mean():.1f}")
                with c3:
                    st.markdown("**Utilization & Cost**")
                    st.markdown(f"- Office visits: {grp['UTIL_OFFICE'].mean():.0f}/yr")
                    st.markdown(f"- ER visits: {grp['UTIL_ER'].mean():.1f}/yr")
                    st.markdown(f"- Rx fills: {grp['UTIL_RX'].mean():.0f}/yr")
                    st.markdown(f"- Total spending: ${grp['EXP_TOTAL'].mean():,.0f}")

        ## Intervention mapping
        st.markdown("---")
        st.markdown("## Suggested Interventions")
        st.markdown(
            "Based on each persona's clinical profile and utilization patterns, "
            "different interventions are recommended:"
        )

        intervention_map = {}
        for c in cluster_ids:
            grp = merged[merged["CLUSTER"] == c]
            persona = grp["PERSONA"].iloc[0]
            interventions = []
            if grp["N_CHRONIC"].mean() >= 2:
                interventions.append("Care coordination")
            if grp["UTIL_RX"].mean() > 30:
                interventions.append("Medication therapy management")
            if grp["UTIL_ER"].mean() > 0.5:
                interventions.append("ER diversion program")
            if grp["UTIL_INPAT"].mean() > 0.3:
                interventions.append("Transitional care planning")
            if grp["DX_DIABETES"].mean() > 0.3:
                interventions.append("Diabetes management")
            if grp["PHYS_HEALTH"].mean() >= 3.5:
                interventions.append("Chronic disease self-management")
            if grp["INS_PUBLIC"].mean() > 0.6:
                interventions.append("Benefits navigation")
            if grp["AGE"].mean() >= 65:
                interventions.append("Medicare wellness screening")
            if not interventions:
                interventions.append("General preventive outreach")
            intervention_map[c] = {"persona": persona, "interventions": interventions}

        for c in cluster_ids:
            info = intervention_map[c]
            st.markdown(f"#### Cluster {int(c) + 1}: {info['persona'][:40]}")
            for intv in info["interventions"]:
                st.markdown(f"- {intv}")
            st.markdown("")


# =============================================================================
# PAGE 5: ROI SIMULATOR
# =============================================================================
elif page == "ROI Simulator":

    st.markdown("# Return on Investment Simulator")
    st.markdown(
        "Adjust the assumptions below to explore how the financial impact changes "
        "under different scenarios. All projections are scaled to a target population size."
    )

    st.markdown("---")

    ## Assumption controls
    st.markdown("## Assumptions")
    st.markdown("*Drag the sliders to test different scenarios.*")

    col1, col2 = st.columns(2)
    with col1:
        population = st.slider("Population Size", 10_000, 500_000, 100_000, step=10_000,
                               help="Total members in the health plan")
        intervention_cost = st.slider("Intervention Cost ($/member/year)", 500, 6000, 2000, step=100,
                                      help="Cost of enrolling one member in a care management program")
        overhead = st.slider("Program Fixed Overhead ($/year)", 0, 200_000, 50_000, step=10_000,
                             help="Annual fixed cost for program staff, IT, and admin")
    with col2:
        reduction_rate = st.slider("Cost Reduction Rate (%)", 1, 40, 15,
                                   help="Expected % reduction in next-year spending for true high-cost patients who receive intervention") / 100
        capacity_pct = st.slider("Targeting Capacity (%)", 1, 30, 10,
                                 help="What % of the population can be enrolled in the program")
        inflation_rate = st.slider("Annual Healthcare Inflation (%)", 0, 10, 3,
                                   help="Expected annual increase in healthcare costs — inflates the projected avoided costs") / 100

    st.markdown("---")

    ## Run simulation
    if data["preds"] is not None:
        preds = data["preds"]
        y_test = preds["HIGH_COST"].values
        actual_cost = preds["TOTEXP_NEXT"].values

        proba_col = [c for c in preds.columns if "xgboost" in c and "calibrated" in c]
        if not proba_col:
            proba_col = [c for c in preds.columns if c.startswith("proba_")]
        y_proba = preds[proba_col[-1]].values

        n_sample = len(y_test)
        scale = population / n_sample

        # Apply inflation to projected costs
        inflated_cost = actual_cost * (1 + inflation_rate)

        # Model-targeted
        k = max(1, int(n_sample * capacity_pct / 100))
        model_idx = np.argsort(y_proba)[::-1][:k]
        model_hc = y_test[model_idx].sum()
        model_gross = inflated_cost[model_idx][y_test[model_idx] == 1].sum() * reduction_rate * scale
        model_cost = (k * scale * intervention_cost + overhead)
        model_net = model_gross - model_cost
        model_recall = model_hc / y_test.sum()

        # Random
        rng = np.random.RandomState(42)
        rand_results = []
        for _ in range(200):
            rand_idx = rng.choice(n_sample, k, replace=False)
            rand_hc = y_test[rand_idx].sum()
            rand_gross = inflated_cost[rand_idx][y_test[rand_idx] == 1].sum() * reduction_rate * scale
            rand_net = rand_gross - model_cost
            rand_results.append(rand_net)
        random_net = np.mean(rand_results)
        random_recall = k / n_sample  # approx

        ## Key results
        st.markdown("## Results")

        roi_pct = f"{model_net / model_cost * 100:.0f}%" if model_cost > 0 else "N/A"

        row1_col1, row1_col2 = st.columns(2)
        row1_col1.metric("Members Targeted", f"{int(k * scale):,}")
        row1_col2.metric("High-Cost Patients Captured", f"{model_recall:.0%}")

        row2_col1, row2_col2 = st.columns(2)
        row2_col1.metric("Net Savings (Model)", f"${model_net:,.0f}",
                          delta=f"${model_net - random_net:,.0f} vs random")
        row2_col2.metric("Program ROI", roi_pct)

        st.markdown(f"""
        <div class="callout-navy">
            <p><strong>At {capacity_pct}% capacity</strong> with ${intervention_cost:,}/member 
            and {reduction_rate:.0%} cost reduction, the model-targeted program generates 
            <strong>${model_net:,.0f}</strong> in net savings for a {population:,}-member plan.
            Random targeting at the same capacity would {"save" if random_net > 0 else "lose"} 
            <strong>${abs(random_net):,.0f}</strong>.
            The model advantage is <strong>${model_net - random_net:,.0f}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        ## Capacity sweep chart
        st.markdown("## Savings Across Capacity Levels")

        cap_range = list(range(1, 31))
        model_nets_sweep = []
        random_nets_sweep = []

        for cap in cap_range:
            kk = max(1, int(n_sample * cap / 100))
            idx = np.argsort(y_proba)[::-1][:kk]
            mg = inflated_cost[idx][y_test[idx] == 1].sum() * reduction_rate * scale
            mc = kk * scale * intervention_cost + overhead
            model_nets_sweep.append(mg - mc)

            # Random approximation
            expected_hc = kk * y_test.mean()
            avg_hc_cost = inflated_cost[y_test == 1].mean()
            rg = expected_hc * avg_hc_cost * reduction_rate * scale
            random_nets_sweep.append(rg - mc)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cap_range, y=model_nets_sweep, mode="lines",
            name="Model-Targeted", line=dict(color=COLORS["terracotta"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=cap_range, y=random_nets_sweep, mode="lines",
            name="Random Outreach", line=dict(color=COLORS["warm_gray"], width=2, dash="dash"),
        ))
        fig.add_hline(y=0, line_color="#888", line_width=0.5)
        fig.add_vline(x=capacity_pct, line_color=COLORS["terracotta"], line_width=1, line_dash="dot",
                      annotation_text=f"Current: {capacity_pct}%", annotation_position="top")
        fig.update_layout(
            **PLOT_LAYOUT, height=400,
            title=f"Net Savings by Targeting Capacity — {population:,}-Member Plan",
            xaxis_title="% of Population Targeted",
            yaxis_title="Net Savings ($)",
            legend=dict(x=0.6, y=0.95, font=dict(size=12, color="#1a2332")),
        )
        fig.update_yaxes(gridcolor="#bbb", tickformat="$,.0f")
        fix_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        ## Persona-based ROI breakdown
        if data["clusters"] is not None:
            st.markdown("---")
            st.markdown("## Persona-Based ROI Breakdown")
            st.markdown(
                "Different patient segments respond differently to intervention. "
                "Persona-tailored targeting applies higher effectiveness rates where evidence is strongest."
            )

            clusters = data["clusters"]
            persona_rates = {0: 0.10, 1: 0.12, 2: 0.22, 3: 0.18}

            # Compute persona-based savings
            persona_results = {}
            for c in sorted(clusters["CLUSTER"].unique()):
                c_dupers = set(clusters[clusters["CLUSTER"] == c]["DUPERSID"])
                c_targeted = [i for i in model_idx if preds.iloc[i]["DUPERSID"] in c_dupers]

                if len(c_targeted) == 0:
                    continue

                c_arr = np.array(c_targeted)
                c_hc = y_test[c_arr].sum()
                rate = persona_rates.get(c, reduction_rate)
                c_gross = inflated_cost[c_arr][y_test[c_arr] == 1].sum() * rate * scale
                c_cost = len(c_targeted) * scale * intervention_cost
                c_net = c_gross - c_cost

                persona_name = clusters[clusters["CLUSTER"] == c]["PERSONA"].iloc[0]
                persona_results[c] = {
                    "name": persona_name, "n": len(c_targeted), "hc": int(c_hc),
                    "rate": rate, "gross": c_gross, "cost": c_cost, "net": c_net,
                }

            if persona_results:
                persona_net_total = sum(r["net"] for r in persona_results.values())

                st.markdown(f"""
                <div class="callout-box">
                    <strong>Flat model targeting</strong> (uniform {reduction_rate:.0%}): Net savings = <strong>${model_net:,.0f}</strong><br>
                    <strong>Persona-tailored targeting</strong> (variable rates): Net savings = <strong>${persona_net_total:,.0f}</strong><br>
                    <strong>Persona advantage: ${persona_net_total - model_net:,.0f}</strong>
                </div>
                """, unsafe_allow_html=True)

                # Bar chart: net savings by persona
                fig = go.Figure()
                c_names = [f"C{int(c)+1}: {persona_results[c]['name'][:25]}" for c in persona_results]
                c_nets = [persona_results[c]["net"] for c in persona_results]
                c_rates_list = [persona_results[c]["rate"] for c in persona_results]

                fig.add_trace(go.Bar(
                    x=c_names, y=c_nets,
                    marker_color=[COLORS["sage"] if n > 0 else COLORS["error"] for n in c_nets],
                    text=[f"{r:.0%} eff." for r in c_rates_list],
                    textposition="outside",
                ))
                fig.add_hline(y=0, line_color="#888", line_width=0.5)
                fig.update_layout(
                    **PLOT_LAYOUT, height=400,
                    title="Net Savings by Persona (Tailored Effectiveness Rates)",
                    yaxis_title="Net Savings ($)", yaxis_tickformat="$,.0f",
                    showlegend=False,
                )
                fix_axes(fig)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="callout-box">
                    <strong>Key insight:</strong> Cluster 3 (Chronic Rx-Driven) generates the vast majority of savings.
                    A smart health plan should concentrate care management investment on this segment
                    and apply lighter-touch monitoring to the other groups.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        ## Sensitivity analysis
        st.markdown("## Sensitivity Analysis")
        st.markdown("How do results change if our assumptions are wrong?")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Varying Cost Reduction Rate")
            rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            rate_nets = []
            for r in rates:
                mg = inflated_cost[model_idx][y_test[model_idx] == 1].sum() * r * scale
                rate_nets.append(mg - model_cost)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"{r:.0%}" for r in rates], y=rate_nets,
                marker_color=[COLORS["sage"] if n > 0 else COLORS["error"] for n in rate_nets],
            ))
            fig.add_hline(y=0, line_color="#888", line_width=0.5)
            fig.update_layout(**PLOT_LAYOUT, height=350, yaxis_tickformat="$,.0f",
                              xaxis_title="Cost Reduction Rate", yaxis_title="Net Savings",
                              showlegend=False)
            fix_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Varying Intervention Cost")
            costs = [500, 1000, 1500, 2000, 3000, 4000, 5000]
            cost_nets = []
            for c in costs:
                mc = k * scale * c + overhead
                cost_nets.append(model_gross - mc)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"${c:,}" for c in costs], y=cost_nets,
                marker_color=[COLORS["sage"] if n > 0 else COLORS["error"] for n in cost_nets],
            ))
            fig.add_hline(y=0, line_color="#888", line_width=0.5)
            fig.update_layout(**PLOT_LAYOUT, height=350, yaxis_tickformat="$,.0f",
                              xaxis_title="Cost per Member", yaxis_title="Net Savings",
                              showlegend=False)
            fix_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

        ## Assumptions documentation
        st.markdown("---")
        st.markdown("## Assumptions & Caveats")
        st.markdown("""
        <div class="callout-box">
            <strong>This is a simulation, not a guarantee.</strong> Key caveats include:
            <ul>
                <li>The cost reduction rate is assumed, not measured from a clinical trial</li>
                <li>MEPS is survey data — an actual health plan's population may differ</li>
                <li>No regression-to-the-mean adjustment has been applied</li>
                <li>The model identifies <em>associations</em>, not causal relationships</li>
                <li>Intervention effectiveness likely varies by patient; persona-specific rates would improve accuracy</li>
            </ul>
            All assumptions are adjustable via the sliders above.
        </div>
        """, unsafe_allow_html=True)
