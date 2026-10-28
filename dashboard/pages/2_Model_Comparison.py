"""Model Comparison Page: Cross-model accuracy, efficiency, and minority class comparisons."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Comparison - KD Citation", page_icon="⚖️", layout="wide")

st.title("⚖️ Comprehensive Model Comparison")
st.markdown("Quantitative evaluation comparing **SciBERT Teacher** against **Supervised Baselines** and **Distilled Students**.")

results_dir = Path("results")
bench_file = results_dir / "benchmarks_summary.json"
manifest_file = results_dir / "manifest.json"

models_data = []
if bench_file.exists():
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_json = json.load(f)
        models_data = bench_json.get("models", [])
elif manifest_file.exists():
    with open(manifest_file, "r", encoding="utf-8") as f:
        man_json = json.load(f)
        models_data = man_json.get("experiments", [])

if not models_data:
    st.info("No benchmark results found yet. Experiments are currently running or haven't executed `scripts/run_benchmarks.py`.")
    st.stop()

df = pd.DataFrame(models_data)

# Filter Controls in Expander
with st.expander("🔍 Filter Models", expanded=False):
    f_cols = st.columns(3)
    methods = df["method"].unique().tolist() if "method" in df.columns else []
    with f_cols[0]:
        sel_methods = st.multiselect("Distillation Method", methods, default=methods)
    with f_cols[1]:
        min_f1 = st.slider("Minimum Macro-F1", 0.0, 1.0, 0.0, 0.05)
    with f_cols[2]:
        max_params = st.number_input("Max Parameters (Millions)", min_value=1.0, max_value=150.0, value=150.0)

filtered_df = df.copy()
if "method" in filtered_df.columns and sel_methods:
    filtered_df = filtered_df[filtered_df["method"].isin(sel_methods)]
if "test_macro_f1" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["test_macro_f1"] >= min_f1]
if "total_params" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["total_params"] <= (max_params * 1e6)]

# Comparative Table
st.subheader("📋 Leaderboard & Performance Metrics")

display_cols = [
    "model_alias", "method", "test_macro_f1", "test_accuracy", "minority_f1",
    "total_params", "checkpoint_size_mb", "latency_single_ms", "throughput_single_eps"
]
available_cols = [c for c in display_cols if c in filtered_df.columns]

st.dataframe(
    filtered_df[available_cols].rename(columns={
        "model_alias": "Model",
        "method": "Distillation Method",
        "test_macro_f1": "Macro-F1",
        "test_accuracy": "Accuracy",
        "minority_f1": "Result F1 (Minority)",
        "total_params": "Total Params",
        "checkpoint_size_mb": "Checkpoint (MB)",
        "latency_single_ms": "Latency (ms)",
        "throughput_single_eps": "Throughput (ex/s)",
    }),
    use_container_width=True,
)

st.divider()

# Comparative Visualizations
col_c1, col_c2 = st.columns(2)

with col_c1:
    if "test_macro_f1" in filtered_df.columns and "minority_f1" in filtered_df.columns:
        melted_df = filtered_df.melt(
            id_vars=["model_alias"],
            value_vars=["test_macro_f1", "minority_f1"],
            var_name="Metric",
            value_name="Score"
        )
        melted_df["Metric"] = melted_df["Metric"].replace({
            "test_macro_f1": "Overall Macro-F1",
            "minority_f1": "Minority F1 ('Result')"
        })
        fig_bar = px.bar(
            melted_df,
            x="model_alias",
            y="Score",
            color="Metric",
            barmode="group",
            title="Macro-F1 vs Minority-Class ('Result') Performance",
            color_discrete_sequence=["#3B82F6", "#F59E0B"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with col_c2:
    if "latency_single_ms" in filtered_df.columns and "total_params" in filtered_df.columns:
        fig_lat = px.bar(
            filtered_df,
            x="model_alias",
            y="latency_single_ms",
            color="method",
            title="Inference Latency by Model Architecture (Lower is Better)",
            labels={"latency_single_ms": "Median Latency (ms)", "model_alias": "Model"},
        )
        st.plotly_chart(fig_lat, use_container_width=True)

st.divider()

# Multi-metric Radar Chart
st.subheader("🕸️ Multi-Dimensional Capability Radar")
if len(filtered_df) > 0 and "test_macro_f1" in filtered_df.columns:
    categories = ["Macro-F1", "Minority F1", "Param Compression", "Speedup"]

    fig_radar = go.Figure()

    # Find teacher baseline for normalization
    teacher_row = df[df["model_type"] == "teacher"]
    t_params = teacher_row["total_params"].values[0] if not teacher_row.empty else 110_000_000
    t_lat = teacher_row["latency_single_ms"].values[0] if (not teacher_row.empty and "latency_single_ms" in teacher_row) else 50.0

    for _, row in filtered_df.iterrows():
        f1_val = row.get("test_macro_f1", 0.0)
        min_f1_val = row.get("minority_f1", 0.0)
        params_ratio = min(1.0, t_params / (row.get("total_params", t_params) + 1e-6) / 20.0)  # normalized compression score
        speedup = min(1.0, (t_lat / (row.get("latency_single_ms", t_lat) + 1e-6)) / 5.0) if "latency_single_ms" in row else 0.5

        scores = [f1_val, min_f1_val, params_ratio, speedup]
        scores.append(scores[0])  # close polygon

        fig_radar.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories + [categories[0]],
            fill="toself",
            name=row.get("model_alias", "Model"),
            opacity=0.6,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Radar Representation (Scores Scaled 0 to 1)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
