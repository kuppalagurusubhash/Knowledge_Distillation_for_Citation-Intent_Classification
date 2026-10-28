"""Training Experiments Page: Loss trajectories, epoch metrics, and convergence logs."""
import glob
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Training Experiments", page_icon="📈", layout="wide")

st.title("📈 Training Dynamics & Experiment Tracking")
st.markdown("Inspect training and validation loss curves, macro-F1 convergence trajectories, and execution histories.")

results_dir = Path("results")
history_files = list(results_dir.glob("*_history.csv"))

if not history_files:
    st.info("No training history logs found yet. Once models finish training epochs, trajectories will appear here.")
    st.stop()

# Experiment Selector
exp_options = [f.stem.replace("_history", "") for f in history_files]
selected_exp = st.selectbox("Select Experiment to Inspect:", exp_options)

selected_file = results_dir / f"{selected_exp}_history.csv"
metrics_file = results_dir / f"{selected_exp}_metrics.json"

df_history = pd.read_csv(selected_file)

# Summary Info
meta_dict = {}
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

final_m = meta_dict.get("final_metrics", {})
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Epochs", f"{len(df_history)}")
with c2:
    best_f1 = df_history["val_macro_f1"].max() if "val_macro_f1" in df_history.columns else 0.0
    st.metric("Best Val Macro-F1", f"{best_f1:.4f}")
with c3:
    final_acc = final_m.get("test_metrics", {}).get("accuracy", "N/A")
    st.metric("Test Accuracy", f"{final_acc}")
with c4:
    train_time = final_m.get("training_time_seconds", "N/A")
    st.metric("Training Time", f"{train_time}s" if train_time != "N/A" else "N/A")

st.divider()

# Loss & Metric Curves
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    fig_loss = go.Figure()
    if "train_loss" in df_history.columns:
        fig_loss.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["train_loss"],
            mode="lines+markers", name="Train Loss", line=dict(color="#3B82F6", width=2.5)
        ))
    if "val_loss" in df_history.columns:
        fig_loss.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_loss"],
            mode="lines+markers", name="Val Loss", line=dict(color="#EF4444", width=2.5, dash="dash")
        ))
    fig_loss.update_layout(
        title=f"Loss Trajectory across Epochs ({selected_exp})",
        xaxis_title="Epoch",
        yaxis_title="Cross-Entropy / Distillation Loss",
    )
    st.plotly_chart(fig_loss, use_container_width=True)

with col_plot2:
    fig_f1 = go.Figure()
    if "val_macro_f1" in df_history.columns:
        fig_f1.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_macro_f1"],
            mode="lines+markers", name="Validation Macro-F1", line=dict(color="#10B981", width=2.5)
        ))
    if "val_minority_f1" in df_history.columns:
        fig_f1.add_trace(go.Scatter(
            x=df_history["epoch"], y=df_history["val_minority_f1"],
            mode="lines+markers", name="Minority ('Result') F1", line=dict(color="#F59E0B", width=2.5, dash="dot")
        ))
    fig_f1.update_layout(
        title=f"Validation F1 Progression ({selected_exp})",
        xaxis_title="Epoch",
        yaxis_title="F1 Score",
    )
    st.plotly_chart(fig_f1, use_container_width=True)

# History Data Table
st.subheader("📊 Epoch History Table")
st.dataframe(df_history, use_container_width=True)

# Metadata & Configuration Inspector
with st.expander("🛠️ Inspect Saved Experiment Metadata & Config", expanded=False):
    st.json(meta_dict)
