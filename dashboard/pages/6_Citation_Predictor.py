"""Citation Intent Predictor Page: Live real-time inference on trained model checkpoints."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st

st.set_page_config(page_title="Citation Intent Predictor", page_icon="🤖", layout="wide")

st.title("🤖 Live Scientific Citation-Intent Predictor")
st.markdown("Test citation sentences against **real trained model checkpoints** to observe predicted intent and confidence scores.")

ID2LABEL = {0: "Method", 1: "Background", 2: "Result"}
LABEL_DESCRIPTIONS = {
    "Method": "The cited paper is utilized as a tool, algorithm, dataset, or methodology.",
    "Background": "The cited paper provides contextual foundation, related work, or theoretical background.",
    "Result": "The current findings are directly compared, contrasted, or evaluated against the cited work.",
}

# Discover available trained checkpoints
checkpoints_root = Path("artifacts/checkpoints")
available_ckpts = {}
if checkpoints_root.exists():
    for d in checkpoints_root.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            available_ckpts[d.name] = str(d)

if not available_ckpts:
    st.warning("No trained checkpoints detected in `artifacts/checkpoints/`. Please train at least one model first.")
    st.stop()

# Model Selector
selected_model_name = st.selectbox(
    "Select Trained Checkpoint to Run:",
    list(available_ckpts.keys()),
    index=0,
)
selected_ckpt_path = available_ckpts[selected_model_name]

# Load model cache
@st.cache_resource(show_spinner="Loading model checkpoint...")
def get_predictor_model(ckpt_path: str):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.eval()
    return model, tokenizer

model, tokenizer = get_predictor_model(selected_ckpt_path)

st.divider()

# Sample Citations for Quick Testing
st.subheader("💡 Or Select a Real Sample Citation Sentence:")
sample_options = {
    "Select or type below...": "",
    "Method Example": "We preprocess the input sequences using the BPE algorithm proposed by Sennrich et al. [24] with 32,000 merge operations.",
    "Background Example": "Transformer architectures have demonstrated unprecedented performance across a multitude of NLP benchmarks (Vaswani et al., 2017).",
    "Result Comparison Example": "Our proposed knowledge distillation student outperforms the baseline model reported by Devlin et al. [14] by 2.4% in macro-F1.",
}
sample_choice = st.selectbox("Pre-filled Scientific Examples:", list(sample_options.keys()))

default_text = sample_options[sample_choice] if sample_choice != "Select or type below..." else ""

# Text Input Area
input_sentence = st.text_area(
    "Scientific Citation Sentence:",
    value=default_text,
    placeholder="Paste a sentence from an academic paper with an in-text citation...",
    height=120,
)

btn_predict = st.button("🚀 Classify Citation Intent", type="primary")

if btn_predict or input_sentence.strip():
    if not input_sentence.strip():
        st.warning("Please enter or select a citation sentence.")
    else:
        # Benchmark single-sentence inference
        t0 = time.perf_counter()
        encoded = tokenizer(
            input_sentence,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL.get(pred_id, "Unknown")
        confidence = probs[pred_id] * 100.0

        st.subheader("🎯 Prediction Results")
        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            color_badge = {
                "Method": "#10B981",
                "Background": "#3B82F6",
                "Result": "#F59E0B"
            }.get(pred_label, "#64748B")

            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border-radius: 10px; padding: 1.5rem; border-left: 6px solid {color_badge};">
                    <h3 style="margin: 0; color: #1E293B;">Predicted Intent:</h3>
                    <h1 style="color: {color_badge}; margin: 0.3rem 0;">{pred_label}</h1>
                    <p style="font-size: 1.1rem; color: #475569; margin-bottom: 0.5rem;">Confidence: <b>{confidence:.2f}%</b></p>
                    <p style="font-size: 0.95rem; color: #64748B;">{LABEL_DESCRIPTIONS.get(pred_label, '')}</p>
                    <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #E2E8F0;">
                    <p style="font-size: 0.85rem; color: #94A3B8; margin: 0;">Inference latency: <b>{latency_ms:.2f} ms</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            df_probs = pd.DataFrame({
                "Intent": [ID2LABEL[i] for i in range(3)],
                "Probability (%)": [round(float(p) * 100, 2) for p in probs],
            })
            fig_probs = px.bar(
                df_probs,
                x="Probability (%)",
                y="Intent",
                orientation="h",
                text="Probability (%)",
                title="Class Probability Distribution",
                color="Intent",
                color_discrete_map={"Background": "#3B82F6", "Method": "#10B981", "Result": "#F59E0B"},
            )
            fig_probs.update_layout(xaxis=dict(range=[0, 100]), showlegend=False)
            st.plotly_chart(fig_probs, use_container_width=True)
