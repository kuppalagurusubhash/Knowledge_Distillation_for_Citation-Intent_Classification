"""Pareto Analysis Page: Multi-objective trade-offs between accuracy, latency, and model size."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pareto Analysis - KD Citation", page_icon="📉", layout="wide")

st.title("📉 Multi-Objective Pareto Frontier Analysis")
st.markdown(
    "Explore the **Pareto-optimal frontiers** balancing classification accuracy against computational footprint, "
    "inference latency, and hardware throughput."
)

bench_file = Path("results/benchmarks_summary.json")

if not bench_file.exists():
    st.info("No benchmark results found yet. Run `python scripts/run_benchmarks.py` after models have finished training.")
    st.stop()

with open(bench_file, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

models = bench_data.get("models", [])
if not models:
    st.info("No model benchmark records available.")
    st.stop()

df = pd.DataFrame(models)
df["params_millions"] = df["total_params"] / 1e6

pareto_size_names = bench_data.get("pareto_size_vs_f1_names", [])
pareto_lat_names = bench_data.get("pareto_latency_vs_f1_names", [])

# Highlights
st.subheader("🏆 Pareto-Optimal Frontier Models")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 📏 Optimal under Model Size vs. Macro-F1")
    for name in pareto_size_names:
        st.markdown(f"- ✅ **{name}** (Pareto-efficient)")
with c2:
    st.markdown("#### ⚡ Optimal under Latency vs. Macro-F1")
    for name in pareto_lat_names:
        st.markdown(f"- 🚀 **{name}** (Pareto-efficient)")

st.divider()

# Interactive Pareto Plots
col_p1, col_p2 = st.columns(2)

with col_p1:
    # Macro-F1 vs Model Size
    fig_size = px.scatter(
        df,
        x="params_millions",
        y="test_macro_f1",
        color="method",
        size="throughput_single_eps",
        hover_name="model_alias",
        hover_data=["test_accuracy", "minority_f1", "latency_single_ms"],
        title="Accuracy vs Model Size (Bubble size = Throughput)",
        labels={"params_millions": "Parameters (Millions)", "test_macro_f1": "Test Macro-F1"},
    )
    # Add Pareto line connecting size-efficient models sorted by size
    size_eff_df = df[df["is_pareto_size_vs_f1"]].sort_values("params_millions")
    if len(size_eff_df) > 1:
        fig_size.add_trace(go.Scatter(
            x=size_eff_df["params_millions"],
            y=size_eff_df["test_macro_f1"],
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="#10B981", width=2, dash="dash"),
        ))
    st.plotly_chart(fig_size, use_container_width=True)

with col_p2:
    # Macro-F1 vs Latency
    fig_lat = px.scatter(
        df,
        x="latency_single_ms",
        y="test_macro_f1",
        color="method",
        size="params_millions",
        hover_name="model_alias",
        hover_data=["test_accuracy", "minority_f1", "throughput_single_eps"],
        title="Accuracy vs Median Latency (Bubble size = Parameters)",
        labels={"latency_single_ms": "Single Example Latency (ms)", "test_macro_f1": "Test Macro-F1"},
    )
    lat_eff_df = df[df["is_pareto_latency_vs_f1"]].sort_values("latency_single_ms")
    if len(lat_eff_df) > 1:
        fig_lat.add_trace(go.Scatter(
            x=lat_eff_df["latency_single_ms"],
            y=lat_eff_df["test_macro_f1"],
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="#3B82F6", width=2, dash="dash"),
        ))
    st.plotly_chart(fig_lat, use_container_width=True)

st.divider()

# Computational Budget Recommendation Tool
st.subheader("🎯 Operating Budget Recommendation System")
st.markdown("Set your computational limits to identify the optimal model operating point:")

b_col1, b_col2 = st.columns(2)
with b_col1:
    budget_lat = st.slider("Maximum Permissible Latency (ms):", 1.0, 100.0, 30.0, 1.0)
with b_col2:
    budget_params = st.slider("Maximum Model Parameter Footprint (Millions):", 1.0, 120.0, 50.0, 5.0)

eligible = df[(df["latency_single_ms"] <= budget_lat) & (df["params_millions"] <= budget_params)]
if not eligible.empty:
    best_candidate = eligible.sort_values("test_macro_f1", ascending=False).iloc[0]
    st.success(
        f"🌟 **Recommended Optimal Model**: **{best_candidate['model_alias']}** (`{best_candidate['method']}`)\n\n"
        f"- **Macro-F1**: {best_candidate['test_macro_f1']:.4f}\n"
        f"- **Latency**: {best_candidate['latency_single_ms']:.2f} ms\n"
        f"- **Footprint**: {best_candidate['params_millions']:.1f}M params ({best_candidate['checkpoint_size_mb']} MB)\n"
        f"- **Throughput**: {best_candidate['throughput_single_eps']:.1f} citations/sec"
    )
else:
    st.warning("No evaluated models meet the current strict budget thresholds. Try relaxing latency or parameter limits.")
