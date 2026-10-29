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
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .intent-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.1rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    .intent-method {
        border-left-color: #10B981 !important;
    }
    .intent-background {
        border-left-color: #3B82F6 !important;
    }
    .intent-result {
        border-left-color: #F59E0B !important;
    }
    .badge-completed {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-method {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-background {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-result {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
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
    "**Course**: CAIML_Satheesha_2_PS_114\n\n"
    "**Faculty / Mentor**: Dr. T.Y. Satheesha\n\n"
    "**Benchmark**: SciCite (8,194 citations)\n\n"
    "**Teacher**: SciBERT (`allenai/scibert_scivocab_uncased`)\n\n"
    "**Students**: DistilBERT, BERT-Mini, BERT-Tiny\n\n"
    "**Verified Models**: 8 Completed Checkpoints"
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
best_macro_f1 = max([e.get("test_macro_f1", 0.0) for e in experiments], default=0.0)
fastest_latency = min([e.get("latency_single_ms", 999.0) for e in experiments if "latency_single_ms" in e], default=0.0)
smallest_size = min([e.get("total_params", 0) for e in experiments], default=0)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Evaluated Models", f"{num_completed} Verified", delta="8 Completed", delta_color="normal")
with col2:
    st.metric("Reference Teacher", "SciBERT (110M)", delta="F1: 0.8655", delta_color="off")
with col3:
    st.metric("Best Verified Macro-F1", f"{best_macro_f1:.4f}", delta="DistilBERT KD (+0.0029 vs Teacher)", delta_color="normal")
with col4:
    st.metric("Smallest Student", f"{smallest_size / 1e6:.1f}M params", delta="96% Smaller (BERT-Tiny)", delta_color="inverse")
with col5:
    st.metric("Fastest CPU Latency", f"{fastest_latency:.2f} ms", delta="50x Faster (vs Teacher 143.6ms)", delta_color="normal")

st.divider()

# Section 1.1: Project Objective & Citation Intent Schema
st.subheader("🎯 1. Project Objective & Citation-Intent Schema")

col_obj, col_classes = st.columns([1, 1])

with col_obj:
    st.markdown("#### 🔬 Research Problem & Motivation")
    st.markdown(
        """
        In academic literature, citations indicate how new discoveries relate to prior work. 
        However, classifying citations at scale using large scientific language models like **SciBERT** 
        (109.9M parameters, ~420 MB checkpoint) creates severe computational bottlenecks: high inference 
        latency (~144 ms/citation on CPU) and high memory footprint make it impractical for real-time document search engines, 
        edge reference managers, or large-scale bibliometric mapping.
        
        **This research investigates Knowledge Distillation (KD)** to compress SciBERT into compact student 
        architectures (**DistilBERT, BERT-Mini, BERT-Tiny**) across multiple objectives (logit matching, intermediate 
        feature projection, and perturbation augmentation). We rigorously evaluate whether compact students preserve 
        **accuracy**, **minority-class recall** (`result`), **computational efficiency**, and **behavioural robustness**.
        """
    )

with col_classes:
    st.markdown("#### 🏷️ The Three Citation-Intent Classes (SciCite)")
    
    st.markdown(
        """
        <div class="intent-card intent-method">
            <span class="badge-method">METHOD (Class 0)</span>
            <p style="margin: 0.3rem 0; font-weight: 500; color: #1E293B;">Tool, algorithm, dataset, library, or experimental protocol adopted from prior work.</p>
            <p style="margin: 0; font-size: 0.85rem; color: #64748B; font-style: italic;">
                Example: "We preprocess input sequences using the BPE algorithm proposed by Sennrich et al. [24] with 32,000 merge operations."
            </p>
        </div>
        
        <div class="intent-card intent-background">
            <span class="badge-background">BACKGROUND (Class 1)</span>
            <p style="margin: 0.3rem 0; font-weight: 500; color: #1E293B;">Foundational context, domain theory, motivation, or related work review.</p>
            <p style="margin: 0; font-size: 0.85rem; color: #64748B; font-style: italic;">
                Example: "Transformer architectures have demonstrated unprecedented performance across a multitude of NLP benchmarks (Vaswani et al., 2017)."
            </p>
        </div>
        
        <div class="intent-card intent-result">
            <span class="badge-result">RESULT COMPARISON (Class 2) — Minority Class (~14%)</span>
            <p style="margin: 0.3rem 0; font-weight: 500; color: #1E293B;">Direct comparison, contrast, confirmation, or discrepancy between experimental findings.</p>
            <p style="margin: 0; font-size: 0.85rem; color: #64748B; font-style: italic;">
                Example: "Our proposed knowledge distillation student outperforms the baseline model reported by Devlin et al. [14] by 2.4% in macro-F1."
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Section 1.2: Model-Size & Efficiency Tiers
st.subheader("⚖️ 2. Architectural Compression & Size Tiers")

st.markdown(
    """
    We systematically benchmark models across **four capacity tiers**, covering progressive levels of parameter and memory reduction:
    """
)

col_t1, col_t2, col_t3, col_t4 = st.columns(4)

with col_t1:
    st.markdown(
        """
        <div style="background: #F1F5F9; border-radius: 8px; padding: 1rem; border-top: 4px solid #475569;">
            <h4 style="margin: 0; color: #1E293B;">🏛️ Teacher Tier</h4>
            <h3 style="margin: 0.3rem 0; color: #0F172A;">SciBERT</h3>
            <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;">allenai/scibert_scivocab_uncased</p>
            <ul style="padding-left: 1.2rem; font-size: 0.85rem; color: #334155; margin: 0;">
                <li><b>Parameters:</b> 109.9M (12 layers)</li>
                <li><b>Checkpoint:</b> 420.2 MB</li>
                <li><b>CPU Latency:</b> 143.6 ms/citation</li>
                <li><b>Macro-F1:</b> 0.8655</li>
                <li><b>Role:</b> Authoritative reference teacher</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_t2:
    st.markdown(
        """
        <div style="background: #F1F5F9; border-radius: 8px; padding: 1rem; border-top: 4px solid #3B82F6;">
            <h4 style="margin: 0; color: #1E293B;">🚀 Compact Student</h4>
            <h3 style="margin: 0.3rem 0; color: #1D4ED8;">DistilBERT</h3>
            <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;">distilbert-base-uncased</p>
            <ul style="padding-left: 1.2rem; font-size: 0.85rem; color: #334155; margin: 0;">
                <li><b>Parameters:</b> 66.9M (6 layers, -39%)</li>
                <li><b>Checkpoint:</b> 256.3 MB</li>
                <li><b>CPU Latency:</b> 68.6 ms (2.1x faster)</li>
                <li><b>Macro-F1:</b> 0.8684 (KD) / 0.8335 (Sup)</li>
                <li><b>Role:</b> High-accuracy production sweet spot</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_t3:
    st.markdown(
        """
        <div style="background: #F1F5F9; border-radius: 8px; padding: 1rem; border-top: 4px solid #10B981;">
            <h4 style="margin: 0; color: #1E293B;">⚡ Edge Student</h4>
            <h3 style="margin: 0.3rem 0; color: #047857;">BERT-Mini</h3>
            <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;">prajjwal1/bert-mini</p>
            <ul style="padding-left: 1.2rem; font-size: 0.85rem; color: #334155; margin: 0;">
                <li><b>Parameters:</b> 11.2M (4 layers, -90%)</li>
                <li><b>Checkpoint:</b> 43.5 MB</li>
                <li><b>CPU Latency:</b> 8.8 ms (14.1x faster)</li>
                <li><b>Macro-F1:</b> 0.7702 (Sup) / 0.7665 (KD)</li>
                <li><b>Role:</b> Ultra-fast CPU edge classification</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_t4:
    st.markdown(
        """
        <div style="background: #F1F5F9; border-radius: 8px; padding: 1rem; border-top: 4px solid #F59E0B;">
            <h4 style="margin: 0; color: #1E293B;">🔍 Ultra-Small Student</h4>
            <h3 style="margin: 0.3rem 0; color: #B45309;">BERT-Tiny</h3>
            <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;">prajjwal1/bert-tiny</p>
            <ul style="padding-left: 1.2rem; font-size: 0.85rem; color: #334155; margin: 0;">
                <li><b>Parameters:</b> 4.4M (2 layers, -96%)</li>
                <li><b>Checkpoint:</b> 17.6 MB</li>
                <li><b>CPU Latency:</b> 2.9 ms (50.2x faster)</li>
                <li><b>Macro-F1:</b> 0.2328</li>
                <li><b>Role:</b> Highlights compression capacity limit</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Section 1.3: Concise Research Summary & Empirical Findings
st.subheader("💡 3. Concise Research Summary & Key Findings")

st.markdown(
    """
    1. **Knowledge Distillation Closes the Gap**:
       On DistilBERT, soft-label distillation from SciBERT improves Macro-F1 from **0.8335** (supervised) to **0.8684** (+3.49%) 
       and boosts minority `result` F1 from **0.7447** to **0.8043** (+5.96%), surrendering zero accuracy relative to the teacher while halving CPU latency (143.6 ms to 68.6 ms).
    2. **Extreme Efficiency with BERT-Mini**:
       BERT-Mini achieves a **14.1x speedup** (8.8 ms/citation, processing 113.4 citations/second on CPU) while retaining **0.7702** Macro-F1. 
       This provides an ideal operating point for large-scale document parsing.
    3. **The Capacity Collapse Threshold**:
       Extreme compression to BERT-Tiny (4.4M parameters) causes capacity collapse on the low-frequency `result` intent (F1 drops to 0.0), 
       rigorously demonstrating that sub-10M parameter models require specialized structural retention techniques for tail classes.
    4. **Robustness Preservation**:
       Distilled students exhibit superior stability against typographic noise: BERT-Mini KD suffered only **2.27%** degradation under typos 
       compared to **5.60%** for its supervised counterpart, proving that teacher soft labels act as an effective regularizer.
    """
)

st.divider()

# Section 1.4: Verified Model Manifest Table
st.subheader("🧪 4. Verified Model Manifest & Leaderboard")
if experiments:
    df_exp = pd.DataFrame(experiments)
    cols_map = {
        "model_alias": "Model",
        "method": "Training Method",
        "test_macro_f1": "Macro-F1",
        "test_accuracy": "Accuracy",
        "minority_f1": "Result F1 (Minority)",
        "total_params": "Parameters",
        "checkpoint_size_mb": "Checkpoint (MB)",
        "status": "Status",
    }
    avail_cols = [c for c in cols_map.keys() if c in df_exp.columns]
    df_show = df_exp[avail_cols].rename(columns=cols_map)
    st.dataframe(df_show, use_container_width=True)
else:
    st.info("Loading experiment manifest...")

st.divider()

# Section 1.5: Navigation Cards
st.subheader("🧭 5. Interactive Dashboard Modules")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("### 🤖 [Citation Predictor & PDF Analyzer](Citation_Predictor)")
    st.caption("Classify citation sentences interactively and upload research paper PDFs to extract and categorize citations in real time.")
with m2:
    st.markdown("### ⚖️ [Model Comparison](Model_Comparison)")
    st.caption("Deep-dive into Supervised vs. Distillation trade-offs, per-class metrics, latency, and throughput.")
with m3:
    st.markdown("### 📉 [Pareto Analysis](Pareto_Analysis)")
    st.caption("Inspect multi-objective Pareto frontiers: Size vs. Macro-F1 and Latency vs. Macro-F1.")

m4, m5, m6 = st.columns(3)
with m4:
    st.markdown("### 🛡️ [Robustness & Error Analysis](Robustness_Analysis)")
    st.caption("Evaluate resilience against whitespace, punctuation, and typographic perturbations; inspect confusion matrices.")
with m5:
    st.markdown("### 📈 [Training Experiments](Training_Experiments)")
    st.caption("Review epoch loss curves, validation Macro-F1 trajectories, and experiment configurations.")
with m6:
    st.markdown("### 📑 [Research Report & Exports](Research_Report)")
    st.caption("Read the synthesized research paper report and download benchmark CSVs, JSONs, and model configs.")
