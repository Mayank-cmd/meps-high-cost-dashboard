"""
=============================================================================
MEPS High-Cost Patient — Step 6: ROI / Cost-Benefit Simulation
=============================================================================
Team 1 | MISM 6212

Prerequisites:
    pip install pandas numpy matplotlib seaborn

Input:
    ./features_output/test_features.csv
    ./tuned_output/predictions_test_tuned.csv
    ./cluster_output/cluster_assignments.csv

Output:
    ./roi_output/roi_capacity_curve.png           (savings vs capacity)
    ./roi_output/roi_strategy_comparison.png       (model vs random vs persona)
    ./roi_output/roi_sensitivity_analysis.png      (sensitivity to assumptions)
    ./roi_output/roi_persona_breakdown.png          (savings by persona)
    ./roi_output/roi_executive_summary.png          (1-page visual summary)
    ./roi_output/roi_simulation_results.csv         (detailed scenario table)
    ./roi_output/roi_analysis_summary.txt           (full text summary)

Usage:
    python step6_roi_simulation.py

Assumptions (conservative, documented, adjustable):
    - Intervention cost per member per year: $2,000
    - Cost reduction among true high-cost if intervened: 15%
    - No cost reduction for patients incorrectly flagged (false positives)
    - Program fixed overhead: $50,000/year
    These are configurable in the CONFIG section below.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG — Adjust these assumptions for your scenario
# =============================================================================
FEATURES_DIR = "./features_output"
TUNED_DIR = "./tuned_output"
CLUSTER_DIR = "./cluster_output"
OUTPUT_DIR = "./roi_output"
RANDOM_STATE = 42

# --- Intervention assumptions (conservative defaults) ---
INTERVENTION_COST_PER_MEMBER = 2000     # $/year per enrolled member
COST_REDUCTION_RATE = 0.15              # 15% reduction in next-year cost for true HC
PROGRAM_FIXED_OVERHEAD = 50_000         # annual fixed cost (staff, IT, admin)
POPULATION_SCALE = 100_000              # scale to a 100K member health plan

# --- Sensitivity ranges ---
COST_REDUCTION_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
INTERVENTION_COST_RANGE = [500, 1000, 1500, 2000, 3000, 4000]

# --- Persona-specific assumptions ---
# Different personas may respond differently to intervention
# Based on SHAP-based clustering (step5b)
PERSONA_EFFECTIVENESS = {
    0: 0.10,    # Stable Elderly — low current spend, monitoring stage, modest response
    1: 0.12,    # Acute/Inpatient-Driven — episodic costs harder to prevent
    2: 0.22,    # Chronic Rx-Driven — prime target, medication management has evidence
    3: 0.18,    # Poor Health, Not Yet Expensive — early intervention opportunity
}

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

# Load cluster assignments (for persona-based targeting)
try:
    cluster_df = pd.read_csv(os.path.join(CLUSTER_DIR, "cluster_assignments.csv"))
    HAS_CLUSTERS = True
    log(f"Cluster assignments loaded: {len(cluster_df):,} high-risk patients")
except FileNotFoundError:
    HAS_CLUSTERS = False
    log("Cluster assignments not found — skipping persona-based targeting.")

# Identify best model probability column
proba_cols = [c for c in preds_df.columns if c.startswith("proba_")]
best_proba_col = None
for c in proba_cols:
    if "xgboost" in c and "calibrated" in c:
        best_proba_col = c
        break
if best_proba_col is None:
    best_proba_col = proba_cols[-1]

preds_df["PRED_PROB"] = preds_df[best_proba_col]
y_true = preds_df["HIGH_COST"].values
y_proba = preds_df["PRED_PROB"].values
actual_cost = preds_df["TOTEXP_NEXT"].values

n_total = len(preds_df)
n_hc = y_true.sum()
total_spend = actual_cost.sum()
hc_spend = actual_cost[y_true == 1].sum()

log(f"\nTest population: {n_total:,}")
log(f"True high-cost: {n_hc} ({n_hc/n_total*100:.1f}%)")
log(f"Total spending: ${total_spend:,.0f}")
log(f"High-cost spending: ${hc_spend:,.0f} ({hc_spend/total_spend*100:.1f}% of total)")
log(f"Mean high-cost expenditure: ${actual_cost[y_true==1].mean():,.0f}")


# =============================================================================
# 2. DEFINE TARGETING STRATEGIES
# =============================================================================
log("\n" + "=" * 72)
log("STEP 2: SIMULATION FRAMEWORK")
log("=" * 72)
log(f"""
  Assumptions:
    Intervention cost:      ${INTERVENTION_COST_PER_MEMBER:,}/member/year
    Cost reduction rate:    {COST_REDUCTION_RATE:.0%} (for true high-cost, if intervened)
    Program overhead:       ${PROGRAM_FIXED_OVERHEAD:,}/year
    Population scale:       {POPULATION_SCALE:,} members

  Three targeting strategies:
    1. RANDOM:  Randomly select K members for intervention
    2. MODEL:   Select top K by predicted risk score
    3. PERSONA: Select top K by risk score, tailor intervention by cluster
""")


def simulate_roi(capacity_pct, y_true, y_proba, actual_cost,
                 intervention_cost=INTERVENTION_COST_PER_MEMBER,
                 reduction_rate=COST_REDUCTION_RATE,
                 overhead=PROGRAM_FIXED_OVERHEAD,
                 strategy="model", cluster_labels=None,
                 persona_rates=None, n_simulations=100):
    """
    Simulate ROI for a given capacity (% of population targeted).

    Returns dict with: net_savings, gross_savings, total_cost, recall,
                       precision, n_targeted, n_true_hc_captured
    """
    n = len(y_true)
    k = max(1, int(n * capacity_pct / 100))

    if strategy == "model":
        # Target top-K by predicted risk
        targeted_idx = np.argsort(y_proba)[::-1][:k]

        true_hc_captured = y_true[targeted_idx].sum()
        avoidable_cost = actual_cost[targeted_idx][y_true[targeted_idx] == 1].sum()
        gross_savings = avoidable_cost * reduction_rate
        total_intervention_cost = k * intervention_cost + overhead
        net_savings = gross_savings - total_intervention_cost

        return {
            "n_targeted": k,
            "n_true_hc": true_hc_captured,
            "recall": true_hc_captured / y_true.sum(),
            "precision": true_hc_captured / k,
            "gross_savings": gross_savings,
            "intervention_cost": total_intervention_cost,
            "net_savings": net_savings,
            "roi_pct": (net_savings / total_intervention_cost * 100)
                       if total_intervention_cost > 0 else 0,
        }

    elif strategy == "random":
        # Average over multiple random draws
        results = []
        rng = np.random.RandomState(RANDOM_STATE)
        for _ in range(n_simulations):
            targeted_idx = rng.choice(n, k, replace=False)
            true_hc_captured = y_true[targeted_idx].sum()
            avoidable_cost = actual_cost[targeted_idx][y_true[targeted_idx] == 1].sum()
            gross_savings = avoidable_cost * reduction_rate
            total_intervention_cost = k * intervention_cost + overhead
            net_savings = gross_savings - total_intervention_cost
            results.append({
                "n_targeted": k,
                "n_true_hc": true_hc_captured,
                "recall": true_hc_captured / y_true.sum(),
                "precision": true_hc_captured / k,
                "gross_savings": gross_savings,
                "intervention_cost": total_intervention_cost,
                "net_savings": net_savings,
                "roi_pct": (net_savings / total_intervention_cost * 100)
                           if total_intervention_cost > 0 else 0,
            })
        # Return mean
        return {key: np.mean([r[key] for r in results]) for key in results[0]}

    elif strategy == "persona" and cluster_labels is not None and persona_rates is not None:
        # Same targeting as model, but apply persona-specific reduction rates
        targeted_idx = np.argsort(y_proba)[::-1][:k]

        gross_savings = 0
        for idx in targeted_idx:
            if y_true[idx] == 1:
                # Find this patient's cluster
                if idx in cluster_labels.index:
                    c = cluster_labels.loc[idx]
                    rate = persona_rates.get(c, reduction_rate)
                else:
                    rate = reduction_rate
                gross_savings += actual_cost[idx] * rate

        true_hc_captured = y_true[targeted_idx].sum()
        total_intervention_cost = k * intervention_cost + overhead
        net_savings = gross_savings - total_intervention_cost

        return {
            "n_targeted": k,
            "n_true_hc": true_hc_captured,
            "recall": true_hc_captured / y_true.sum(),
            "precision": true_hc_captured / k,
            "gross_savings": gross_savings,
            "intervention_cost": total_intervention_cost,
            "net_savings": net_savings,
            "roi_pct": (net_savings / total_intervention_cost * 100)
                       if total_intervention_cost > 0 else 0,
        }


# =============================================================================
# 3. RUN CAPACITY SWEEP
# =============================================================================
log("=" * 72)
log("STEP 3: CAPACITY SWEEP (1% to 30%)")
log("=" * 72)

capacity_range = list(range(1, 31))

# Prepare cluster labels indexed by test_df position
if HAS_CLUSTERS:
    dupersid_to_idx = dict(zip(preds_df["DUPERSID"], preds_df.index))
    cluster_series = pd.Series(dtype=int)
    for _, row in cluster_df.iterrows():
        idx = dupersid_to_idx.get(row["DUPERSID"])
        if idx is not None:
            cluster_series[idx] = int(row["CLUSTER"])

strategies = ["model", "random"]
if HAS_CLUSTERS:
    strategies.append("persona")

sweep_results = {s: [] for s in strategies}

for cap in capacity_range:
    for strat in strategies:
        if strat == "persona":
            res = simulate_roi(cap, y_true, y_proba, actual_cost,
                               strategy="persona",
                               cluster_labels=cluster_series,
                               persona_rates=PERSONA_EFFECTIVENESS)
        else:
            res = simulate_roi(cap, y_true, y_proba, actual_cost, strategy=strat)
        res["capacity_pct"] = cap
        sweep_results[strat].append(res)

# Print key results
log(f"\n{'Cap%':>5s} | {'Strategy':<10s} | {'Targeted':>8s} | {'HC Found':>8s} | "
    f"{'Recall':>7s} | {'Gross $':>12s} | {'Cost':>10s} | {'Net $':>12s} | {'ROI%':>7s}")
log("─" * 100)

for cap in [2, 5, 10, 15, 20]:
    for strat in strategies:
        r = sweep_results[strat][cap - 1]
        log(f"{cap:>4}% | {strat:<10s} | {r['n_targeted']:>8,} | {r['n_true_hc']:>8.0f} | "
            f"{r['recall']:>6.1%} | ${r['gross_savings']:>11,.0f} | "
            f"${r['intervention_cost']:>9,.0f} | ${r['net_savings']:>11,.0f} | "
            f"{r['roi_pct']:>6.1f}%")
    log("")


# =============================================================================
# 4. SCALE TO FULL POPULATION
# =============================================================================
log("\n" + "=" * 72)
log(f"STEP 4: SCALED PROJECTIONS ({POPULATION_SCALE:,}-MEMBER HEALTH PLAN)")
log("=" * 72)

scale_factor = POPULATION_SCALE / n_total

log(f"\nScale factor: {scale_factor:.1f}x")
log(f"\n{'Cap%':>5s} | {'Strategy':<10s} | {'Members':>8s} | "
    f"{'Gross Savings':>14s} | {'Program Cost':>13s} | {'Net Savings':>14s} | {'ROI':>7s}")
log("─" * 85)

scaled_results = []
for cap in [5, 10, 15, 20]:
    for strat in ["model", "random"]:
        r = sweep_results[strat][cap - 1]
        scaled = {
            "Capacity (%)": cap,
            "Strategy": strat.title(),
            "Members Targeted": int(r["n_targeted"] * scale_factor),
            "HC Captured": int(r["n_true_hc"] * scale_factor),
            "Recall": r["recall"],
            "Gross Savings ($)": r["gross_savings"] * scale_factor,
            "Program Cost ($)": r["n_targeted"] * scale_factor * INTERVENTION_COST_PER_MEMBER + PROGRAM_FIXED_OVERHEAD,
            "Net Savings ($)": None,  # computed below
            "ROI (%)": None,
        }
        scaled["Net Savings ($)"] = scaled["Gross Savings ($)"] - scaled["Program Cost ($)"]
        scaled["ROI (%)"] = (scaled["Net Savings ($)"] / scaled["Program Cost ($)"] * 100
                              if scaled["Program Cost ($)"] > 0 else 0)
        scaled_results.append(scaled)

        log(f"{cap:>4}% | {strat:<10s} | {scaled['Members Targeted']:>8,} | "
            f"${scaled['Gross Savings ($)']:>13,.0f} | ${scaled['Program Cost ($)']:>12,.0f} | "
            f"${scaled['Net Savings ($)']:>13,.0f} | {scaled['ROI (%)']:>6.1f}%")
    # Compute marginal value of model vs random
    model_r = [s for s in scaled_results if s["Capacity (%)"] == cap and s["Strategy"] == "Model"][-1]
    random_r = [s for s in scaled_results if s["Capacity (%)"] == cap and s["Strategy"] == "Random"][-1]
    incremental = model_r["Net Savings ($)"] - random_r["Net Savings ($)"]
    log(f"        {'Model advantage':>10s}: ${incremental:>13,.0f}")
    log("")


# =============================================================================
# 5. SENSITIVITY ANALYSIS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 5: SENSITIVITY ANALYSIS")
log("=" * 72)

# Fix capacity at 10%, vary reduction rate
log("\nA. Varying cost reduction rate (capacity fixed at 10%):")
sensitivity_reduction = []
for rate in COST_REDUCTION_RANGE:
    r = simulate_roi(10, y_true, y_proba, actual_cost,
                     reduction_rate=rate, strategy="model")
    r_rand = simulate_roi(10, y_true, y_proba, actual_cost,
                          reduction_rate=rate, strategy="random")
    sensitivity_reduction.append({
        "reduction_rate": rate,
        "model_net": r["net_savings"],
        "random_net": r_rand["net_savings"],
        "model_roi": r["roi_pct"],
        "advantage": r["net_savings"] - r_rand["net_savings"],
    })
    log(f"  Rate={rate:.0%}: Model net=${r['net_savings']:>10,.0f} "
        f"Random net=${r_rand['net_savings']:>10,.0f} "
        f"Advantage=${r['net_savings']-r_rand['net_savings']:>10,.0f}")

# Fix capacity at 10%, vary intervention cost
log("\nB. Varying intervention cost per member (capacity fixed at 10%):")
sensitivity_cost = []
for cost in INTERVENTION_COST_RANGE:
    r = simulate_roi(10, y_true, y_proba, actual_cost,
                     intervention_cost=cost, strategy="model")
    sensitivity_cost.append({
        "intervention_cost": cost,
        "net_savings": r["net_savings"],
        "roi_pct": r["roi_pct"],
        "breakeven": r["net_savings"] > 0,
    })
    log(f"  Cost=${cost:>5,}: Net=${r['net_savings']:>10,.0f}  "
        f"ROI={r['roi_pct']:>6.1f}%  "
        f"{'POSITIVE' if r['net_savings'] > 0 else 'NEGATIVE'}")

# Find breakeven intervention cost
log("\nC. Breakeven analysis (capacity=10%, reduction=15%):")
for cost_test in range(100, 10000, 100):
    r = simulate_roi(10, y_true, y_proba, actual_cost,
                     intervention_cost=cost_test, strategy="model")
    if r["net_savings"] <= 0:
        log(f"  Breakeven intervention cost: ~${cost_test - 100:,} - ${cost_test:,} per member")
        break


# =============================================================================
# 6. PERSONA-BASED ROI (if clusters available)
# =============================================================================
if HAS_CLUSTERS:
    log("\n" + "=" * 72)
    log("STEP 6: PERSONA-BASED ROI BREAKDOWN")
    log("=" * 72)

    # For each persona, compute contribution to savings at 20% capacity
    targeted_idx_20 = np.argsort(y_proba)[::-1][:int(n_total * 0.20)]

    persona_roi = {}
    for c in sorted(cluster_df["CLUSTER"].unique()):
        persona_dupers = set(cluster_df[cluster_df["CLUSTER"] == c]["DUPERSID"])
        persona_targeted = [i for i in targeted_idx_20
                            if preds_df.iloc[i]["DUPERSID"] in persona_dupers]

        if len(persona_targeted) == 0:
            continue

        n_targeted = len(persona_targeted)
        targeted_mask = np.array(persona_targeted)
        hc_in_persona = y_true[targeted_mask].sum()
        hc_cost = actual_cost[targeted_mask][y_true[targeted_mask] == 1].sum()
        rate = PERSONA_EFFECTIVENESS.get(c, COST_REDUCTION_RATE)
        gross = hc_cost * rate
        cost_total = n_targeted * INTERVENTION_COST_PER_MEMBER
        net = gross - cost_total

        persona_name = cluster_df[cluster_df["CLUSTER"] == c]["PERSONA"].iloc[0]
        persona_roi[c] = {
            "name": persona_name,
            "n_targeted": n_targeted,
            "n_hc": hc_in_persona,
            "gross_savings": gross,
            "cost": cost_total,
            "net_savings": net,
            "effectiveness": rate,
        }
        log(f"\n  Cluster {int(c)+1}: \"{persona_name}\"")
        log(f"    Targeted: {n_targeted}, True HC: {hc_in_persona}")
        log(f"    Effectiveness: {rate:.0%}")
        log(f"    Gross savings: ${gross:,.0f}")
        log(f"    Cost: ${cost_total:,.0f}")
        log(f"    Net savings: ${net:,.0f}")


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 7: GENERATING VISUALIZATIONS")
log("=" * 72)

strategy_colors = {"model": "#1565C0", "random": "#9E9E9E", "persona": "#E53935"}
strategy_labels = {"model": "Model-Targeted", "random": "Random Outreach", "persona": "Persona-Tailored"}


# ---- FIGURE 1: Net Savings vs Capacity ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROI Analysis — Model-Targeted vs Random Intervention",
             fontsize=14, fontweight="bold")

ax = axes[0]
for strat in strategies:
    net_savings = [r["net_savings"] for r in sweep_results[strat]]
    ax.plot(capacity_range, net_savings,
            label=strategy_labels[strat], color=strategy_colors[strat],
            lw=2, marker="o", ms=3)
ax.axhline(0, color="black", ls="--", alpha=0.3)
ax.set_xlabel("Capacity (% of Population Targeted)")
ax.set_ylabel("Net Savings ($)")
ax.set_title("Net Savings by Targeting Strategy")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Shade the advantage region
model_net = [r["net_savings"] for r in sweep_results["model"]]
random_net = [r["net_savings"] for r in sweep_results["random"]]
ax.fill_between(capacity_range, model_net, random_net,
                alpha=0.15, color="#1565C0", label="Model advantage")

ax = axes[1]
for strat in strategies:
    recalls = [r["recall"] * 100 for r in sweep_results[strat]]
    ax.plot(capacity_range, recalls,
            label=strategy_labels[strat], color=strategy_colors[strat],
            lw=2, marker="o", ms=3)
ax.plot(capacity_range, capacity_range, "k--", alpha=0.3, label="Random baseline")
ax.set_xlabel("Capacity (% of Population Targeted)")
ax.set_ylabel("% of True High-Cost Captured (Recall)")
ax.set_title("Recall by Targeting Strategy")
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roi_capacity_curve.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved roi_capacity_curve.png")


# ---- FIGURE 2: Strategy Comparison (bar chart at key capacities) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Strategy Comparison — Scaled to {POPULATION_SCALE:,}-Member Plan",
             fontsize=14, fontweight="bold")

cap_levels = [5, 10, 15, 20]
x = np.arange(len(cap_levels))
width = 0.35

# Net savings
ax = axes[0]
model_nets = [sweep_results["model"][c-1]["net_savings"] * scale_factor for c in cap_levels]
random_nets = [sweep_results["random"][c-1]["net_savings"] * scale_factor for c in cap_levels]
ax.bar(x - width/2, model_nets, width, label="Model-Targeted", color="#1565C0", alpha=0.85)
ax.bar(x + width/2, random_nets, width, label="Random", color="#9E9E9E", alpha=0.85)
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}%" for c in cap_levels])
ax.set_xlabel("Capacity")
ax.set_ylabel("Net Savings ($)")
ax.set_title("Net Savings (Scaled)")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Incremental value
ax = axes[1]
incremental = [m - r for m, r in zip(model_nets, random_nets)]
ax.bar(x, incremental, 0.5, color="#43A047", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}%" for c in cap_levels])
ax.set_xlabel("Capacity")
ax.set_ylabel("Incremental Value of Model ($)")
ax.set_title("Model Advantage over Random Targeting")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

for i, v in enumerate(incremental):
    ax.text(i, v + max(incremental) * 0.02, f"${v:,.0f}", ha="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roi_strategy_comparison.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved roi_strategy_comparison.png")


# ---- FIGURE 3: Sensitivity Analysis ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Sensitivity Analysis (Capacity = 10%)", fontsize=14, fontweight="bold")

# Reduction rate sensitivity
ax = axes[0]
rates = [s["reduction_rate"] for s in sensitivity_reduction]
model_nets = [s["model_net"] for s in sensitivity_reduction]
random_nets = [s["random_net"] for s in sensitivity_reduction]
ax.plot(rates, model_nets, "o-", color="#1565C0", lw=2, label="Model-Targeted")
ax.plot(rates, random_nets, "o-", color="#9E9E9E", lw=2, label="Random")
ax.axhline(0, color="black", ls="--", alpha=0.3)
ax.fill_between(rates, model_nets, random_nets, alpha=0.15, color="#1565C0")
ax.set_xlabel("Assumed Cost Reduction Rate")
ax.set_ylabel("Net Savings ($)")
ax.set_title("Sensitivity to Cost Reduction Rate")
ax.legend()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Intervention cost sensitivity
ax = axes[1]
costs = [s["intervention_cost"] for s in sensitivity_cost]
nets = [s["net_savings"] for s in sensitivity_cost]
rois = [s["roi_pct"] for s in sensitivity_cost]
colors = ["#43A047" if n > 0 else "#E53935" for n in nets]
ax.bar(range(len(costs)), nets, color=colors, alpha=0.85)
ax.set_xticks(range(len(costs)))
ax.set_xticklabels([f"${c:,}" for c in costs])
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Intervention Cost per Member")
ax.set_ylabel("Net Savings ($)")
ax.set_title("Sensitivity to Intervention Cost")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Add ROI labels
for i, (n, r) in enumerate(zip(nets, rois)):
    ax.text(i, n + abs(max(nets)) * 0.03, f"ROI: {r:.0f}%", ha="center", fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roi_sensitivity_analysis.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved roi_sensitivity_analysis.png")


# ---- FIGURE 4: Persona ROI Breakdown ----
if HAS_CLUSTERS and persona_roi:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Persona-Based ROI Breakdown (at 20% Capacity)",
                 fontsize=14, fontweight="bold")

    cluster_colors_list = ["#1565C0", "#E53935", "#43A047", "#FF8F00", "#7B1FA2"]

    # Savings by persona
    ax = axes[0]
    names = [f"C{c+1}" for c in persona_roi]
    gross = [persona_roi[c]["gross_savings"] for c in persona_roi]
    costs = [persona_roi[c]["cost"] for c in persona_roi]
    net = [persona_roi[c]["net_savings"] for c in persona_roi]

    x = np.arange(len(names))
    ax.bar(x - 0.2, gross, 0.35, label="Gross Savings", color="#43A047", alpha=0.8)
    ax.bar(x + 0.2, costs, 0.35, label="Program Cost", color="#E53935", alpha=0.8)
    ax.set_xticks(x)
    labels = [f"C{c+1}\n{persona_roi[c]['name'][:20]}" for c in persona_roi]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Dollars ($)")
    ax.set_title("Gross Savings vs Cost by Persona")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Net savings by persona
    ax = axes[1]
    bar_colors = ["#43A047" if n > 0 else "#E53935" for n in net]
    ax.bar(x, net, 0.5, color=bar_colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Net Savings ($)")
    ax.set_title("Net Savings by Persona")
    ax.axhline(0, color="black", lw=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    for i, v in enumerate(net):
        ax.text(i, v + abs(max(net)) * 0.03, f"${v:,.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roi_persona_breakdown.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log("  Saved roi_persona_breakdown.png")


# ---- FIGURE 5: Executive Summary (1-page visual) ----
fig = plt.figure(figsize=(16, 10))
fig.suptitle("High-Cost Patient Early Warning — Executive ROI Summary",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Key metric at 10% capacity
r_model_10 = sweep_results["model"][9]
r_random_10 = sweep_results["random"][9]

# Panel 1: Key numbers
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
metrics_text = (
    f"Targeting Top 10% of Members\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"Members targeted:  {int(r_model_10['n_targeted'] * scale_factor):,}\n"
    f"True HC captured:  {int(r_model_10['n_true_hc'] * scale_factor):,}\n"
    f"Recall:            {r_model_10['recall']:.1%}\n"
    f"Precision:         {r_model_10['precision']:.1%}\n\n"
    f"Gross savings:     ${r_model_10['gross_savings'] * scale_factor:,.0f}\n"
    f"Program cost:      ${r_model_10['intervention_cost'] * scale_factor:,.0f}\n"
    f"Net savings:       ${r_model_10['net_savings'] * scale_factor:,.0f}\n"
    f"ROI:               {r_model_10['roi_pct']:.0f}%\n\n"
    f"vs Random:         +${(r_model_10['net_savings'] - r_random_10['net_savings']) * scale_factor:,.0f}"
)
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.8))

# Panel 2: Net savings curve
ax = fig.add_subplot(gs[0, 1:])
for strat in ["model", "random"]:
    net_s = [r["net_savings"] * scale_factor for r in sweep_results[strat]]
    ax.plot(capacity_range, net_s,
            label=strategy_labels[strat], color=strategy_colors[strat], lw=2)
ax.axhline(0, color="black", ls="--", alpha=0.3)
ax.fill_between(capacity_range,
                [r["net_savings"] * scale_factor for r in sweep_results["model"]],
                [r["net_savings"] * scale_factor for r in sweep_results["random"]],
                alpha=0.15, color="#1565C0")
ax.axvline(10, color="#E53935", ls=":", alpha=0.5, label="10% capacity")
ax.set_xlabel("Capacity (%)")
ax.set_ylabel("Net Savings ($)")
ax.set_title(f"Net Savings Curve — {POPULATION_SCALE:,}-Member Plan")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Panel 3: Recall curve
ax = fig.add_subplot(gs[1, 0])
for strat in ["model", "random"]:
    recalls = [r["recall"] * 100 for r in sweep_results[strat]]
    ax.plot(capacity_range, recalls,
            label=strategy_labels[strat], color=strategy_colors[strat], lw=2)
ax.plot(capacity_range, capacity_range, "k--", alpha=0.3)
ax.axvline(10, color="#E53935", ls=":", alpha=0.5)
ax.set_xlabel("Capacity (%)")
ax.set_ylabel("Recall (%)")
ax.set_title("High-Cost Capture Rate")
ax.legend(fontsize=8)

# Panel 4: Sensitivity
ax = fig.add_subplot(gs[1, 1])
rates = [s["reduction_rate"] for s in sensitivity_reduction]
advantages = [s["advantage"] * scale_factor for s in sensitivity_reduction]
ax.bar(range(len(rates)), advantages, color="#43A047", alpha=0.85)
ax.set_xticks(range(len(rates)))
ax.set_xticklabels([f"{r:.0%}" for r in rates])
ax.set_xlabel("Cost Reduction Rate")
ax.set_ylabel("Model Advantage over Random ($)")
ax.set_title("Sensitivity: Model Value by Reduction Rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

# Panel 5: Assumptions box
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
assumptions_text = (
    f"KEY ASSUMPTIONS\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"Cost per member:    ${INTERVENTION_COST_PER_MEMBER:,}/yr\n"
    f"Cost reduction:     {COST_REDUCTION_RATE:.0%}\n"
    f"Program overhead:   ${PROGRAM_FIXED_OVERHEAD:,}/yr\n"
    f"Population:         {POPULATION_SCALE:,}\n\n"
    f"CAVEATS\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"• Associations, not causal\n"
    f"• Survey data (MEPS) may\n"
    f"  differ from plan data\n"
    f"• Reduction rate assumed\n"
    f"  uniform within persona\n"
    f"• No regression to mean\n"
    f"  adjustment applied"
)
ax.text(0.05, 0.95, assumptions_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", alpha=0.8))

plt.savefig(os.path.join(OUTPUT_DIR, "roi_executive_summary.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
log("  Saved roi_executive_summary.png")


# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================
log("\n" + "=" * 72)
log("STEP 8: SAVING OUTPUTS")
log("=" * 72)

# Save detailed results table
results_rows = []
for strat in strategies:
    for r in sweep_results[strat]:
        row = r.copy()
        row["strategy"] = strat
        row["net_savings_scaled"] = r["net_savings"] * scale_factor
        results_rows.append(row)

results_csv = pd.DataFrame(results_rows)
results_csv.to_csv(os.path.join(OUTPUT_DIR, "roi_simulation_results.csv"), index=False)
log("  Saved roi_simulation_results.csv")

# Save summary
with open(os.path.join(OUTPUT_DIR, "roi_analysis_summary.txt"), "w") as f:
    f.write("\n".join(log_lines))
log("  Saved roi_analysis_summary.txt")

log(f"\n{'='*72}")
log("ALL DONE! ROI simulation complete.")
log(f"Outputs in {OUTPUT_DIR}/")
log(f"{'='*72}")