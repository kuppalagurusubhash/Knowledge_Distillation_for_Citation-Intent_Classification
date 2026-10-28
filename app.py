"""Main entrypoint for the Knowledge Distillation Citation Intent Streamlit Dashboard."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="KD Citation-Intent Research Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern academic styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-completed {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-pending {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🎓 Knowledge Distillation for Citation-Intent Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Empirical Investigation of Accuracy–Efficiency Trade-offs & Robustness Preservation in Scientific NLP</div>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("🔬 Research Navigation")
st.sidebar.info(
    "**Project**: KD for Citation-Intent\n\n"
    "**Dataset**: SciCite (AI2)\n\n"
    "**Teacher**: SciBERT (110M)\n\n"
    "**Students**: DistilBERT, BERT-Mini, BERT-Tiny"
)

# Load real artifacts
results_dir = Path("results")
manifest_path = results_dir / "manifest.json"
dataset_path = results_dir / "dataset_summary.json"
benchmarks_path = results_dir / "benchmarks_summary.json"

manifest = {"experiments": []}
if manifest_path.exists():
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        pass

experiments = manifest.get("experiments", [])
num_completed = len([e for e in experiments if e.get("status") == "completed"])

# Top Level KPI Metrics
col1, col2, col3, col4, col5 = st.columns(5)

best_macro_f1 = max([e.get("test_macro_f1", 0.0) for e in experiments], default=0.0)
fastest_latency = min([e.get("latency_single_ms", 999.0) for e in experiments if "latency_single_ms" in e], default=0.0)

with col1:
    st.metric("Total Completed Experiments", f"{num_completed}")

with col2:
    st.metric("Teacher Model", "SciBERT (110M)")

with col3:
    st.metric("Best Verified Macro-F1", f"{best_macro_f1:.4f}" if best_macro_f1 > 0 else "Pending")

with col4:
    smallest_size = min([e.get("total_params", 0) for e in experiments], default=0)
    st.metric("Smallest Student", f"{smallest_size / 1e6:.1f}M params" if smallest_size > 0 else "4.4M (Tiny)")

with col5:
    st.metric("Fastest Median Latency", f"{fastest_latency:.2f} ms" if fastest_latency > 0 and fastest_latency < 900 else "Benchmarking...")

st.divider()

# Overview Sections
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📌 Project Abstract & Research Questions")
    st.markdown(
        """
        Deploying high-capacity scientific transformer models like **SciBERT** in production or real-time bibliometric systems
        is restricted by high memory footprint and inference latency. This research framework systematically compresses
        domain-adapted scientific transformers into lightweight students using **Knowledge Distillation (KD)**.
        
        #### Core Scientific Research Inquiries:
        1. **Compression vs. Semantic Retention**: Can compact students preserve scientific citation intent representation?
        2. **Minority Class Resilience**: Does soft-label distillation alleviate performance degradation on low-frequency classes (`result`)?
        3. **Pareto Optimality**: Which student model architecture and distillation objective maximize accuracy under strict latency budgets?
        4. **Perturbation Robustness**: Are distilled compact models robust against typographic errors and bracket shifts in citations?
        """
    )

with col_right:
    st.subheader("🧪 Completed Experiment Manifest")
    if experiments:
        df_exp = pd.DataFrame(experiments)
        display_cols = [c for c in ["model_alias", "method", "test_macro_f1", "test_accuracy", "minority_f1", "status"] if c in df_exp.columns]
        st.dataframe(df_exp[display_cols], use_container_width=True)
    else:
        st.info("Training pipeline currently executing. Experiments will appear here upon completion.")

st.subheader("🧭 Interactive Dashboard Modules")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("### 📊 [Dataset Explorer](Dataset_Explorer)")
    st.caption("Inspect SciCite split statistics, class distributions, and sample citations.")
with m2:
    st.markdown("### ⚖️ [Model Comparison](Model_Comparison)")
    st.caption("Compare Teacher vs. Supervised vs. Distilled models across all metrics.")
with m3:
    st.markdown("### 📉 [Pareto Analysis](Pareto_Analysis)")
    st.caption("Multi-objective frontiers: Accuracy vs. Size, Latency vs. Macro-F1.")
with m4:
    st.markdown("### 🛡️ [Robustness](Robustness_Analysis)")
    st.caption("Clean vs. perturbed test degradation and syntactic resilience.")

m5, m6, m7, m8 = st.columns(4)
with m5:
    st.markdown("### 📈 [Training Curves](Training_Experiments)")
    st.caption("Epoch loss trajectories, validation metrics, and convergence logs.")
with m6:
    st.markdown("### 🤖 [Citation Predictor](Citation_Predictor)")
    st.caption("Live inference tester: test citation sentences on real trained checkpoints.")
with m7:
    st.markdown("### 📑 [Research Report](Research_Report)")
    st.caption("Full academic report and CSV/JSON data export center.")
with m8:
    st.markdown("### ⚙️ Architecture")
    st.caption("SciBERT Teacher (110M) -> DistilBERT (66M) / BERT-Mini (11M) / BERT-Tiny (4.4M).")
