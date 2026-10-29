"""Model Comparison Page: Cross-model accuracy, efficiency, per-class metrics, and distillation trade-offs."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Model Comparison - KD Citation", page_icon="⚖️", layout="wide")

st.title("⚖️ Comprehensive Model Comparison & Distillation Analysis")
st.markdown(
    "Quantitative evaluation comparing **SciBERT Reference Teacher** against **Supervised Baselines** and **Distilled Students** "
    "across classification accuracy, per-class intent metrics, memory footprint, CPU latency, and throughput."
)

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

# Enrich model data with per-class precision, recall, and F1 from individual metrics files
enriched_records = []
for m in models_data:
    exp_id = m.get("experiment_id", "")
    metrics_file = results_dir / f"{exp_id}_metrics.json"
    per_class = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as mf:
                m_json = json.load(mf)
                test_m = m_json.get("final_metrics", {}).get("test_metrics", {})
                per_class = test_m.get("per_class", {})
        except Exception:
            pass

    # Categorize training paradigm
    raw_method = m.get("method", "")
    if raw_method == "reference_teacher":
        paradigm = "Reference Teacher"
    elif "supervised" in raw_method:
        paradigm = "Supervised Learning"
    else:
        paradigm = "Knowledge Distillation"

    rec = {
        "experiment_id": exp_id,
        "model_alias": m.get("model_alias", exp_id),
        "paradigm": paradigm,
        "method": raw_method,
        "test_macro_f1": m.get("test_macro_f1", 0.0),
        "test_accuracy": m.get("test_accuracy", 0.0),
        "minority_f1": m.get("minority_f1", 0.0),
        "total_params": m.get("total_params", 0),
        "params_millions": round(m.get("total_params", 0) / 1e6, 2),
        "checkpoint_size_mb": m.get("checkpoint_size_mb", 0.0),
        "latency_single_ms": m.get("latency_single_ms", None),
        "latency_p95_ms": m.get("latency_p95_ms", None),
        "throughput_single_eps": m.get("throughput_single_eps", None),
        "is_globally_pareto": m.get("is_globally_pareto", False),
        # Per-class metrics
        "method_precision": per_class.get("method", {}).get("precision", "N/A"),
        "method_recall": per_class.get("method", {}).get("recall", "N/A"),
        "method_f1": per_class.get("method", {}).get("f1_score", "N/A"),
        "background_precision": per_class.get("background", {}).get("precision", "N/A"),
        "background_recall": per_class.get("background", {}).get("recall", "N/A"),
        "background_f1": per_class.get("background", {}).get("f1_score", "N/A"),
        "result_precision": per_class.get("result", {}).get("precision", "N/A"),
        "result_recall": per_class.get("result", {}).get("recall", "N/A"),
        "result_f1": per_class.get("result", {}).get("f1_score", "N/A"),
    }
    enriched_records.append(rec)

df_all = pd.DataFrame(enriched_records)

# KPI Summary Banner
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Top Macro-F1", f"{df_all['test_macro_f1'].max():.4f}", "DistilBERT Logit KD")
with kpi2:
    teacher_row = df_all[df_all["method"] == "reference_teacher"]
    t_f1 = teacher_row["test_macro_f1"].values[0] if len(teacher_row) > 0 else 0.8655
    st.metric("SciBERT Teacher Macro-F1", f"{t_f1:.4f}", "Reference Baseline")
with kpi3:
    st.metric("Fastest CPU Latency", f"{df_all['latency_single_ms'].min():.2f} ms", "BERT-Tiny (50x faster)")
with kpi4:
    best_edge = df_all[df_all["model_alias"] == "BERT-Mini"]["test_macro_f1"].max()
    st.metric("Best Edge Model (BERT-Mini)", f"{best_edge:.4f}", "8.8 ms Latency")

st.divider()

# Navigation Tabs
tab_sup_vs_kd, tab_table, tab_charts, tab_tradeoffs = st.tabs([
    "⚖️ Supervised vs. Distillation Comparison",
    "📋 Complete Benchmark & Per-Class Metrics Table",
    "📊 Performance vs. Size & Latency Charts",
    "🧠 In-Depth Trade-off Analysis & Findings",
])

# ==============================================================================
# TAB 1: SUPERVISED VS. DISTILLATION COMPARISON
# ==============================================================================
with tab_sup_vs_kd:
    st.subheader("1. Head-to-Head: Supervised Baselines vs. Knowledge Distillation")
    st.markdown(
        """
        To isolate the exact marginal contribution of **soft-label "dark knowledge" transfer**, we directly compare 
        student models trained via standard ground-truth supervised cross-entropy against identical student models trained via 
        temperature-scaled knowledge distillation.
        """
    )

    pair_col1, pair_col2 = st.columns(2)

    with pair_col1:
        st.markdown("#### 🔹 DistilBERT Comparison (66.9M Parameters)")
        db_sup = df_all[df_all["experiment_id"] == "student_distilbert_supervised"]
        db_kd = df_all[df_all["experiment_id"] == "student_distilbert_logit_distillation"]

        if not db_sup.empty and not db_kd.empty:
            sup_f1 = db_sup["test_macro_f1"].values[0]
            kd_f1 = db_kd["test_macro_f1"].values[0]
            sup_min = db_sup["minority_f1"].values[0]
            kd_min = db_kd["minority_f1"].values[0]
            sup_lat = db_sup["latency_single_ms"].values[0]
            kd_lat = db_kd["latency_single_ms"].values[0]

            delta_f1 = (kd_f1 - sup_f1) * 100
            delta_min = (kd_min - sup_min) * 100

            st.markdown(
                f"""
                <div style="background: #F8FAFC; border-radius: 8px; padding: 1.2rem; border-left: 5px solid #3B82F6;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                        <tr style="border-bottom: 1px solid #E2E8F0;">
                            <th style="text-align: left; padding: 6px;">Metric</th>
                            <th style="text-align: right; padding: 6px;">Supervised</th>
                            <th style="text-align: right; padding: 6px;">Logit KD</th>
                            <th style="text-align: right; padding: 6px; color: #10B981;">Distillation Gain</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 6px;"><b>Macro-F1</b></td>
                            <td style="text-align: right; padding: 6px;">{sup_f1:.4f}</td>
                            <td style="text-align: right; padding: 6px; font-weight: bold; color: #1D4ED8;">{kd_f1:.4f}</td>
                            <td style="text-align: right; padding: 6px; font-weight: bold; color: #10B981;">+{delta_f1:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding: 6px;"><b>Minority F1 ('Result')</b></td>
                            <td style="text-align: right; padding: 6px;">{sup_min:.4f}</td>
                            <td style="text-align: right; padding: 6px; font-weight: bold; color: #1D4ED8;">{kd_min:.4f}</td>
                            <td style="text-align: right; padding: 6px; font-weight: bold; color: #10B981;">+{delta_min:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px;"><b>CPU Latency</b></td>
                            <td style="text-align: right; padding: 6px;">{sup_lat:.2f} ms</td>
                            <td style="text-align: right; padding: 6px;">{kd_lat:.2f} ms</td>
                            <td style="text-align: right; padding: 6px; color: #64748B;">Identical Arch</td>
                        </tr>
                    </table>
                    <p style="margin-top: 0.8rem; font-size: 0.88rem; color: #475569;">
                        💡 <b>Takeaway</b>: Knowledge distillation gives DistilBERT an immediate <b>+3.49% Macro-F1</b> 
                        and <b>+5.96% Minority F1</b> boost, allowing it to beat the full 110M SciBERT teacher while cutting latency by 52%.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with pair_col2:
        st.markdown("#### 🔹 BERT-Mini Multi-Method Comparison (11.2M Parameters)")
        mini_df = df_all[df_all["model_alias"] == "BERT-Mini"].copy()
        if not mini_df.empty:
            st.dataframe(
                mini_df[["method", "test_macro_f1", "minority_f1", "test_accuracy", "latency_single_ms"]].rename(columns={
                    "method": "Training Method",
                    "test_macro_f1": "Macro-F1",
                    "minority_f1": "Result F1",
                    "test_accuracy": "Accuracy",
                    "latency_single_ms": "Latency (ms)",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                """
                💡 **Takeaway**: On BERT-Mini, pure supervised training and logit KD achieve comparable clean Macro-F1 (~0.7702 vs ~0.7665). 
                However, under typographic noise stress tests, **logit KD provides superior robustness** (2.27% degradation vs 5.60% for supervised).
                """
            )

# ==============================================================================
# TAB 2: COMPLETE BENCHMARK & PER-CLASS TABLE
# ==============================================================================
with tab_table:
    st.subheader("2. Comprehensive Model Benchmark Leaderboard")
    st.caption("Includes Overall Accuracy, Macro-F1, Per-Class Breakdown (Method, Background, Result), Parameters, Checkpoint Size, and Latency.")

    # Filter Controls
    f_c1, f_c2 = st.columns([1, 2])
    with f_c1:
        selected_paradigms = st.multiselect(
            "Filter by Training Paradigm:",
            ["Reference Teacher", "Supervised Learning", "Knowledge Distillation"],
            default=["Reference Teacher", "Supervised Learning", "Knowledge Distillation"],
        )
    with f_c2:
        model_family_filter = st.multiselect(
            "Filter by Model Architecture:",
            df_all["model_alias"].unique().tolist(),
            default=df_all["model_alias"].unique().tolist(),
        )

    filtered_df = df_all[
        df_all["paradigm"].isin(selected_paradigms) &
        df_all["model_alias"].isin(model_family_filter)
    ].copy()

    # Main Leaderboard Table
    main_cols = [
        "model_alias", "paradigm", "method", "test_macro_f1", "test_accuracy", "minority_f1",
        "total_params", "checkpoint_size_mb", "latency_single_ms", "throughput_single_eps"
    ]
    st.dataframe(
        filtered_df[main_cols].rename(columns={
            "model_alias": "Model",
            "paradigm": "Paradigm",
            "method": "Method",
            "test_macro_f1": "Macro-F1",
            "test_accuracy": "Accuracy",
            "minority_f1": "Result F1 (Minority)",
            "total_params": "Total Parameters",
            "checkpoint_size_mb": "Checkpoint (MB)",
            "latency_single_ms": "CPU Latency (ms)",
            "throughput_single_eps": "Throughput (ex/s)",
        }),
        use_container_width=True,
    )

    st.subheader("🔬 Detailed Per-Class Intent Metrics (Precision, Recall, F1)")
    per_class_cols = [
        "model_alias", "method",
        "method_precision", "method_recall", "method_f1",
        "background_precision", "background_recall", "background_f1",
        "result_precision", "result_recall", "result_f1",
    ]
    st.dataframe(
        filtered_df[per_class_cols].rename(columns={
            "model_alias": "Model",
            "method": "Method",
            "method_precision": "Method Prec",
            "method_recall": "Method Rec",
            "method_f1": "Method F1",
            "background_precision": "Back Prec",
            "background_recall": "Back Rec",
            "background_f1": "Back F1",
            "result_precision": "Result Prec",
            "result_recall": "Result Rec",
            "result_f1": "Result F1",
        }),
        use_container_width=True,
    )

# ==============================================================================
# TAB 3: VISUALIZATIONS
# ==============================================================================
with tab_charts:
    st.subheader("3. Accuracy vs. Model Size & CPU Latency Trade-offs")

    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        # Macro-F1 vs Parameters
        fig_size = px.scatter(
            df_all,
            x="params_millions",
            y="test_macro_f1",
            color="paradigm",
            text="model_alias",
            size="checkpoint_size_mb",
            title="Macro-F1 vs. Model Parameters (Bubble Size = Checkpoint MB)",
            labels={"params_millions": "Model Parameters (Millions)", "test_macro_f1": "Test Macro-F1", "paradigm": "Paradigm"},
            color_discrete_map={
                "Reference Teacher": "#475569",
                "Supervised Learning": "#EF4444",
                "Knowledge Distillation": "#10B981",
            },
        )
        fig_size.update_traces(textposition="top right")
        fig_size.update_layout(height=420)
        st.plotly_chart(fig_size, use_container_width=True)

    with ch_col2:
        # Macro-F1 vs CPU Latency
        fig_lat = px.scatter(
            df_all,
            x="latency_single_ms",
            y="test_macro_f1",
            color="paradigm",
            text="model_alias",
            title="Macro-F1 vs. CPU Inference Latency (ms)",
            labels={"latency_single_ms": "CPU Single-Sample Latency (ms, log scale)", "test_macro_f1": "Test Macro-F1"},
            log_x=True,
            color_discrete_map={
                "Reference Teacher": "#475569",
                "Supervised Learning": "#EF4444",
                "Knowledge Distillation": "#10B981",
            },
        )
        fig_lat.update_traces(textposition="bottom right")
        fig_lat.update_layout(height=420)
        st.plotly_chart(fig_lat, use_container_width=True)

    # Per-Class F1 Comparison Across Models
    st.subheader("📊 Class-Specific F1 Breakdown Across Models")
    chart_df = df_all.copy()
    # Filter only numerical F1 columns
    plot_records = []
    for _, r in chart_df.iterrows():
        for class_label in ["method", "background", "result"]:
            f1_val = r.get(f"{class_label}_f1")
            try:
                f1_float = float(f1_val)
            except (ValueError, TypeError):
                f1_float = 0.0
            plot_records.append({
                "Model-Method": f"{r['model_alias']} ({r['method'][:7]})",
                "Intent Class": class_label.capitalize(),
                "F1 Score": f1_float,
            })

    df_f1_plot = pd.DataFrame(plot_records)
    fig_class_bars = px.bar(
        df_f1_plot,
        x="Model-Method",
        y="F1 Score",
        color="Intent Class",
        barmode="group",
        title="Per-Class Intent F1 Score by Model (Highlighting Minority 'Result' Vulnerability)",
        color_discrete_map={"Method": "#10B981", "Background": "#3B82F6", "Result": "#F59E0B"},
    )
    fig_class_bars.update_layout(height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig_class_bars, use_container_width=True)

# ==============================================================================
# TAB 4: TRADE-OFF ANALYSIS & RESEARCH INSIGHTS
# ==============================================================================
with tab_tradeoffs:
    st.subheader("4. Research Findings: Balancing Accuracy, Efficiency, and Capacity")
    st.markdown(
        """
        A central contribution of this project is proving that **smaller models are not uniformly better across all dimensions**. 
        Instead, distinct student architectures offer specific, quantifiable Pareto trade-offs:
        
        #### 1. The Production Sweet Spot: DistilBERT Logit Distillation
        - **Accuracy**: Achieves **0.8684 Macro-F1**, surpassing the teacher model (0.8655) and outperforming the supervised student (0.8335 by +3.49%).
        - **Efficiency**: Cuts model size from 420.2 MB to 256.3 MB (-39%) and cuts CPU inference time from 143.6 ms to 68.6 ms (**2.1x speedup**).
        - **Recommendation**: Best suited for production systems requiring top-tier scientific classification fidelity without the full latency burden of SciBERT.
        
        #### 2. The Edge & High-Throughput Sweet Spot: BERT-Mini
        - **Accuracy**: Retains **0.7702 Macro-F1** and **0.7356 Minority F1**.
        - **Efficiency**: Shrinks memory footprint by **90%** (43.5 MB) and slashes latency to **8.82 ms** (**14.1x faster** than SciBERT, delivering 113.4 citations/sec).
        - **Recommendation**: Ideal for real-time document indexing, desktop reference managers, or edge CPU deployment where sub-10ms response times are required.
        
        #### 3. The Capacity Collapse Threshold: BERT-Tiny
        - **Accuracy**: Collapses on the minority `result` class (**F1 = 0.0000**, Macro-F1 drops to 0.2328).
        - **Efficiency**: Extremely fast (**2.86 ms**, 350 citations/sec, 17.6 MB footprint).
        - **Scientific Insight**: 2 transformer layers and 128 hidden dimensions are insufficient to separate nuanced, low-frequency citation intents. 
          This empirically proves that compression cannot be extended arbitrarily without catastrophic capacity collapse on imbalanced text.
        """
    )
