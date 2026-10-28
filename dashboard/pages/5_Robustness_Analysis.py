"""Robustness Analysis Page: Perturbation resilience and degradation measurements."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Robustness Analysis - KD Citation", page_icon="🛡️", layout="wide")

st.title("🛡️ Robustness Preservation under Perturbations")
st.markdown(
    "Empirical evaluation of student and teacher models on **controlled syntactic, typographic, and bracket perturbations** "
    "to measure whether knowledge distillation preserves semantic robustness."
)

rob_file = Path("results/robustness_summary.json")

if not rob_file.exists():
    st.info("No robustness evaluation results found yet. Run `python scripts/run_robustness.py` after models have finished training.")
    st.stop()

with open(rob_file, "r", encoding="utf-8") as f:
    rob_data = json.load(f)

if not rob_data:
    st.info("No robustness data available.")
    st.stop()

# Build DataFrame
records = []
for exp_id, item in rob_data.items():
    records.append({
        "Model": item.get("model_alias", exp_id),
        "Method": item.get("method", "N/A"),
        "Clean Macro-F1": item.get("clean_macro_f1", 0.0),
        "Robustness Score (%)": item.get("overall_robustness_score", 0.0),
        "Avg Degradation (%)": item.get("average_relative_degradation_pct", 0.0),
        "Clean Minority F1": item.get("clean_minority_f1", 0.0),
    })

df_rob = pd.DataFrame(records)

# Top metrics
c1, c2, c3 = st.columns(3)
with c1:
    best_rob = df_rob.sort_values("Robustness Score (%)", ascending=False).iloc[0]
    st.metric("Most Robust Model", f"{best_rob['Model']}", f"{best_rob['Robustness Score (%)']}% retention")
with c2:
    least_deg = df_rob["Avg Degradation (%)"].min()
    st.metric("Minimum Avg Degradation", f"{least_deg:.2f}%")
with c3:
    avg_clean = df_rob["Clean Macro-F1"].mean()
    st.metric("Average Clean Macro-F1", f"{avg_clean:.4f}")

st.divider()

# Robustness Leaderboard
st.subheader("📋 Robustness Leaderboard across Models")
st.dataframe(df_rob, use_container_width=True)

st.divider()

# Comparative Charts
col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_deg = px.bar(
        df_rob,
        x="Model",
        y="Avg Degradation (%)",
        color="Method",
        title="Average Relative Macro-F1 Degradation (Lower is Better)",
        labels={"Avg Degradation (%)": "Relative Degradation (%)"},
    )
    st.plotly_chart(fig_deg, use_container_width=True)

with col_r2:
    fig_ret = px.bar(
        df_rob,
        x="Model",
        y="Robustness Score (%)",
        color="Method",
        title="Overall Robustness Retention Score (Higher is Better)",
        labels={"Robustness Score (%)": "Retention Score (%)"},
    )
    fig_ret.update_yaxes(range=[50, 105])
    st.plotly_chart(fig_ret, use_container_width=True)

st.divider()

# Per-Perturbation Breakdown
st.subheader("🔬 Breakdown by Perturbation Type")
selected_model_name = st.selectbox("Inspect Perturbation Details for Model:", list(rob_data.keys()))

if selected_model_name in rob_data:
    model_rob = rob_data[selected_model_name]
    perts = model_rob.get("perturbations", {})
    pert_rows = []
    for p_name, p_vals in perts.items():
        pert_rows.append({
            "Perturbation Type": p_name.replace("_", " ").title(),
            "Perturbed Macro-F1": p_vals.get("perturbed_macro_f1", 0.0),
            "Absolute F1 Drop": p_vals.get("absolute_f1_degradation", 0.0),
            "Relative Degradation (%)": p_vals.get("relative_f1_degradation_pct", 0.0),
            "Minority F1 Drop": p_vals.get("minority_f1_degradation", 0.0),
        })
    df_perts = pd.DataFrame(pert_rows)
    st.dataframe(df_perts, use_container_width=True)
