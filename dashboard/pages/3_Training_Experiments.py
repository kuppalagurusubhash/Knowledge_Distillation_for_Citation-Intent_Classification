"""Training Experiments Page: Loss trajectories, epoch metrics, hyperparameters, and convergence logs."""
import glob
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Training Experiments - KD Citation", page_icon="📈", layout="wide")

st.title("📈 Training Dynamics, Hyperparameters & Experiment Tracking")
st.markdown(
    "Inspect training and validation loss trajectories, epoch convergence curves, hyperparameter configurations, "
    "and training execution logs across all 8 experimental models."
)

results_dir = Path("results")
manifest_file = results_dir / "manifest.json"
history_files = list(results_dir.glob("*_history.csv"))

if not history_files:
    st.info("No training history logs found in `results/`.")
    st.stop()

# Load manifest for high-level experiment metadata
manifest_data = {}
if manifest_file.exists():
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

experiments = manifest_data.get("experiments", [])

# Experiment Selector
exp_options = sorted([f.stem.replace("_history", "") for f in history_files])
selected_exp = st.selectbox(
    "Select Experiment to Inspect:",
    exp_options,
    index=exp_options.index("student_distilbert_logit_distillation") if "student_distilbert_logit_distillation" in exp_options else 0,
)

selected_file = results_dir / f"{selected_exp}_history.csv"
metrics_file = results_dir / f"{selected_exp}_metrics.json"

df_history = pd.read_csv(selected_file)

# Summary metadata
meta_dict = {}
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

final_m = meta_dict.get("final_metrics", {})
params_info = final_m.get("parameters", {})
test_m = final_m.get("test_metrics", {})

# Top Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Epochs Trained", f"{len(df_history)}")
with c2:
    best_f1 = df_history["val_macro_f1"].max() if "val_macro_f1" in df_history.columns else 0.0
    st.metric("Best Val Macro-F1", f"{best_f1:.4f}")
with c3:
    final_f1 = test_m.get("macro_f1", "N/A")
    st.metric("Test Macro-F1", f"{final_f1:.4f}" if isinstance(final_f1, float) else f"{final_f1}")
with c4:
    final_acc = test_m.get("accuracy", "N/A")
    st.metric("Test Accuracy", f"{final_acc:.4f}" if isinstance(final_acc, float) else f"{final_acc}")
with c5:
    train_time = final_m.get("training_time_seconds", "N/A")
    st.metric("Training Time", f"{train_time:.1f}s" if isinstance(train_time, (int, float)) else "N/A")

st.divider()

# Experiment Metadata & Hyperparameters Expander
with st.expander("⚙️ View Model Architecture & Hyperparameter Configuration", expanded=True):
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        st.markdown(f"**Model Name / HF Backbone**: `{final_m.get('model_name', 'N/A')}`")
        st.markdown(f"**Model Alias**: `{final_m.get('model_alias', selected_exp)}`")
        st.markdown(f"**Training Method**: `{final_m.get('method', 'N/A')}`")
    with col_cfg2:
        st.markdown(f"**Total Parameters**: `{params_info.get('total_parameters', 0):,}`")
        st.markdown(f"**Parameter Size (MB)**: `{params_info.get('param_size_mb', 0.0)} MB`")
        st.markdown(f"**Checkpoint Size (MB)**: `{final_m.get('checkpoint_size_mb', 0.0)} MB`")
    with col_cfg3:
        st.markdown(f"**Device Used**: `{final_m.get('device', 'cpu').upper()}`")
        st.markdown(f"**Checkpoint Directory**: `{final_m.get('checkpoint_dir', 'N/A')}`")
        st.markdown(f"**Execution Status**: <span style='color: #10B981; font-weight: bold;'>Completed</span>", unsafe_allow_html=True)

st.divider()

# Loss & Metric Convergence Curves
st.subheader("📉 Loss & Metric Trajectories Over Epochs")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    fig_loss = go.Figure()
    if "train_loss" in df_history.columns:
        fig_loss.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["train_loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color="#3B82F6", width=2.5),
        ))
    if "val_loss" in df_history.columns:
        fig_loss.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_loss"],
            mode="lines+markers", name="Val Loss",
            line=dict(color="#EF4444", width=2.5, dash="dash"),
        ))
    fig_loss.update_layout(
        title="Training vs Validation Loss Trajectory",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_loss, use_container_width=True)

with col_plot2:
    fig_metric = go.Figure()
    if "val_macro_f1" in df_history.columns:
        fig_metric.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_macro_f1"],
            mode="lines+markers", name="Val Macro-F1",
            line=dict(color="#10B981", width=2.5),
        ))
    if "val_accuracy" in df_history.columns:
        fig_metric.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_accuracy"],
            mode="lines+markers", name="Val Accuracy",
            line=dict(color="#6366F1", width=2),
        ))
    if "val_minority_f1" in df_history.columns:
        fig_metric.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_minority_f1"],
            mode="lines+markers", name="Val Result F1 (Minority)",
            line=dict(color="#F59E0B", width=2, dash="dot"),
        ))
    fig_metric.update_layout(
        title="Validation Classification Metrics Trajectory",
        xaxis_title="Epoch",
        yaxis_title="Metric Score",
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_metric, use_container_width=True)

# Epoch Log Table
st.subheader("📋 Epoch Log Data Table")
st.dataframe(df_history, use_container_width=True)

csv_exp_history = df_history.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"📥 Download {selected_exp}_history.csv",
    data=csv_exp_history,
    file_name=f"{selected_exp}_history.csv",
    mime="text/csv",
)
