"""Citation Intent Predictor Page: Live real-time inference on single citations and full research paper PDF analysis."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st

from src.utils.pdf_extractor import extract_text_from_pdf, extract_citation_sentences

st.set_page_config(page_title="Citation Intent Predictor & PDF Analyzer", page_icon="🤖", layout="wide")

st.title("🤖 Interactive Citation-Intent Classifier & PDF Paper Analyzer")
st.markdown(
    "Perform live citation-intent classification using **real, verified model checkpoints**. "
    "Classify individual citation sentences or upload an entire scientific research paper PDF to extract and categorize in-text citations."
)

ID2LABEL = {0: "Method", 1: "Background", 2: "Result Comparison"}
LABEL_DESCRIPTIONS = {
    "Method": "The cited paper is utilized as an experimental tool, algorithm, dataset, library, or methodology.",
    "Background": "The cited paper provides contextual foundation, related work review, theoretical basis, or problem motivation.",
    "Result Comparison": "The current paper's experimental findings are directly compared, contrasted, or validated against the cited work.",
}

# Discover available trained checkpoints and build user-friendly display labels
checkpoints_root = Path("artifacts/checkpoints")
available_ckpts = {}
ckpt_labels = {}

friendly_names = {
    "teacher_scibert": "SciBERT — Reference Teacher (109.9M params, 420 MB | Test F1: 0.8655)",
    "student_distilbert_logit_distillation": "DistilBERT — Logit Distillation (66.9M params, 256 MB | Test F1: 0.8684 - Best)",
    "student_distilbert_supervised": "DistilBERT — Supervised Baseline (66.9M params, 256 MB | Test F1: 0.8335)",
    "student_bert_mini_supervised": "BERT-Mini — Supervised Baseline (11.2M params, 43 MB | Test F1: 0.7702)",
    "student_bert_mini_logit_distillation": "BERT-Mini — Logit Distillation (11.2M params, 43 MB | Test F1: 0.7665)",
    "student_bert_mini_augmented_distillation": "BERT-Mini — Augmented Distillation (11.2M params, 43 MB | Test F1: 0.7529)",
    "student_bert_mini_feature_distillation": "BERT-Mini — Feature Distillation (11.2M params, 43 MB | Test F1: 0.7253)",
    "student_bert_tiny_logit_distillation": "BERT-Tiny — Logit Distillation (4.4M params, 17 MB | Test F1: 0.2328)",
}

if checkpoints_root.exists():
    for d in sorted(checkpoints_root.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            label = friendly_names.get(d.name, f"{d.name} (Custom Checkpoint)")
            available_ckpts[label] = str(d)
            ckpt_labels[str(d)] = d.name

if not available_ckpts:
    st.error("No trained checkpoints detected in `artifacts/checkpoints/`. Please ensure checkpoints are available.")
    st.stop()


@st.cache_resource(show_spinner="Loading model checkpoint into memory...")
def load_cached_predictor(ckpt_path: str):
    """Loads tokenizer and model checkpoint safely on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.eval()
    return model, tokenizer


def run_single_inference(text: str, model, tokenizer):
    """Executes single-sentence inference and returns probabilities and latency."""
    t0 = time.perf_counter()
    encoded = tokenizer(
        text,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # Filter inputs to only input_ids and attention_mask for universal model support
    inputs = {k: v for k, v in encoded.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL.get(pred_id, "Unknown")
    confidence = float(probs[pred_id]) * 100.0

    return {
        "pred_id": pred_id,
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": [float(p) for p in probs],
        "latency_ms": latency_ms,
    }


# Tabs for Single Predictor vs PDF Analysis
tab_single, tab_pdf = st.tabs(["✍️ Interactive Single Citation Classifier", "📄 Research Paper PDF Citation Analysis"])

# ==============================================================================
# TAB 1: SINGLE CITATION CLASSIFIER
# ==============================================================================
with tab_single:
    st.subheader("1. Select Trained Model Checkpoint")
    selected_label = st.selectbox(
        "Choose Model for Inference:",
        list(available_ckpts.keys()),
        index=1 if len(available_ckpts) > 1 else 0,  # Default to DistilBERT KD if available
        key="single_model_sel",
    )
    selected_path = available_ckpts[selected_label]
    model, tokenizer = load_cached_predictor(selected_path)

    st.divider()

    st.subheader("2. Input Citation Sentence")
    sample_options = {
        "Custom (Type your own citation sentence below)": "",
        "Method Example": "We preprocess the input sequences using the BPE algorithm proposed by Sennrich et al. [24] with 32,000 merge operations.",
        "Background Example": "Transformer architectures have demonstrated unprecedented performance across a multitude of NLP benchmarks (Vaswani et al., 2017).",
        "Result Comparison Example": "Our proposed knowledge distillation student outperforms the baseline model reported by Devlin et al. [14] by 2.4% in macro-F1.",
        "Method Tool Example": "All models were implemented using PyTorch and Hugging Face Transformers library (Wolf et al., 2020).",
        "Background Domain Example": "Citation intent classification has a long history in bibliometrics and natural language processing (Teufel et al., 2006).",
    }
    sample_choice = st.selectbox("Quick-fill with Verified Scientific Examples:", list(sample_options.keys()))
    default_text = sample_options[sample_choice] if sample_choice != "Custom (Type your own citation sentence below)" else ""

    input_sentence = st.text_area(
        "Scientific Citation Sentence:",
        value=default_text,
        placeholder="Paste a sentence from an academic paper with an in-text citation (e.g., 'We trained our model using the Adam optimizer [14]...')...",
        height=110,
    )

    btn_predict = st.button("🚀 Predict Citation Intent", type="primary", key="btn_single_predict")

    if btn_predict or (input_sentence.strip() and sample_choice != "Custom (Type your own citation sentence below)"):
        if not input_sentence.strip():
            st.warning("Please enter or select a citation sentence.")
        else:
            with st.spinner("Executing real model forward pass..."):
                result = run_single_inference(input_sentence, model, tokenizer)

            pred_label = result["pred_label"]
            confidence = result["confidence"]
            probs = result["probs"]
            latency_ms = result["latency_ms"]

            st.markdown("### 🎯 Classification Results")
            res_col1, res_col2 = st.columns([1, 1])

            color_badge = {
                "Method": "#10B981",
                "Background": "#3B82F6",
                "Result Comparison": "#F59E0B",
            }.get(pred_label, "#64748B")

            with res_col1:
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border-radius: 10px; padding: 1.4rem; border-left: 6px solid {color_badge}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <p style="margin: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Predicted Citation Intent</p>
                        <h2 style="color: {color_badge}; margin: 0.4rem 0;">{pred_label}</h2>
                        <p style="font-size: 1.15rem; color: #1E293B; margin-bottom: 0.6rem;">Confidence Score: <b>{confidence:.2f}%</b></p>
                        <p style="font-size: 0.92rem; color: #475569; line-height: 1.4; margin-bottom: 0.8rem;">{LABEL_DESCRIPTIONS.get(pred_label, '')}</p>
                        <hr style="margin: 0.6rem 0; border: none; border-top: 1px solid #E2E8F0;">
                        <p style="font-size: 0.85rem; color: #64748B; margin: 0;">Model: <b>{ckpt_labels.get(selected_path, 'Active Checkpoint')}</b> &nbsp;|&nbsp; CPU Latency: <b>{latency_ms:.2f} ms</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Input Sentence Verification")
                st.info(f"\"{input_sentence}\"")

            with res_col2:
                df_probs = pd.DataFrame({
                    "Intent": [ID2LABEL[i] for i in range(3)],
                    "Probability (%)": [round(p * 100, 2) for p in probs],
                })
                fig_probs = px.bar(
                    df_probs,
                    x="Probability (%)",
                    y="Intent",
                    orientation="h",
                    text="Probability (%)",
                    title="Model Soft-Probability Distribution",
                    color="Intent",
                    color_discrete_map={"Background": "#3B82F6", "Method": "#10B981", "Result Comparison": "#F59E0B"},
                )
                fig_probs.update_layout(xaxis=dict(range=[0, 100]), showlegend=False, height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_probs, use_container_width=True)

                # Show numerical probability table
                st.dataframe(
                    df_probs.rename(columns={"Intent": "Intent Class", "Probability (%)": "Model Probability (%)"}),
                    use_container_width=True,
                    hide_index=True,
                )

# ==============================================================================
# TAB 2: RESEARCH PAPER PDF CITATION ANALYSIS
# ==============================================================================
with tab_pdf:
    st.subheader("📄 Research Paper PDF Citation Extraction & Analysis")
    st.markdown(
        """
        Upload a digital scientific research paper in PDF format. The extraction engine scans the paper, 
        identifies in-text citation markers (e.g., `[1]`, `[12, 14]`, or `(Author et al., 2020)`), extracts their surrounding 
        context sentences, and classifies their intent using your selected model checkpoint.
        """
    )

    # Disclaimers Box
    st.info(
        "ℹ️ **Important Scope & Accuracy Notice**:\n\n"
        "- **Citation Intent Only**: This feature classifies the functional purpose of citation contexts (Method, Background, Result). "
        "It does **not** verify citation factual correctness, check for plagiarism, validate scientific methodology, or score the quality of the paper.\n"
        "- **Formatting Limitations**: PDF multi-column formatting, section breaks, and formula wrapping may cause sentence boundary shifts. "
        "Extracted citations should be manually reviewed."
    )

    col_upload, col_settings = st.columns([2, 1])

    with col_upload:
        uploaded_pdf = st.file_uploader(
            "Upload Scientific Paper PDF (Text-based):",
            type=["pdf"],
            help="Select a digital scientific research paper PDF (e.g., from arXiv or academic conferences).",
        )

    with col_settings:
        pdf_model_label = st.selectbox(
            "Model for PDF Classification:",
            list(available_ckpts.keys()),
            index=1 if len(available_ckpts) > 1 else 0,
            key="pdf_model_sel",
        )
        pdf_model_path = available_ckpts[pdf_model_label]
        max_citations = st.slider("Max Citations to Extract:", 10, 200, 50, step=10)

    if uploaded_pdf is not None:
        with st.spinner("Extracting text and identifying citation sentences from PDF..."):
            raw_text, pdf_meta = extract_text_from_pdf(uploaded_pdf)

        if pdf_meta["error"]:
            st.error(f"Failed to read PDF file: {pdf_meta['error']}")
        elif pdf_meta["is_scanned"] or pdf_meta["word_count"] < 30:
            st.warning(
                "⚠️ **No readable text could be extracted from this PDF.**\n\n"
                "This document appears to be a scanned image, photographed paper, or protected document. "
                "**Optical Character Recognition (OCR) is not currently implemented; only digital, text-based PDFs are supported.**\n\n"
                "Please upload a digital PDF where text can be highlighted and selected."
            )
        else:
            citation_records = extract_citation_sentences(raw_text, max_sentences=max_citations)

            # Metadata cards
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Total Pages", f"{pdf_meta['page_count']}")
            with p_col2:
                st.metric("Extracted Words", f"{pdf_meta['word_count']:,}")
            with p_col3:
                st.metric("Citation Sentences Found", f"{len(citation_records)}")
            with p_col4:
                st.metric("Active Model", f"{ckpt_labels.get(pdf_model_path, 'Model')[:15]}...")

            if not citation_records:
                st.warning(
                    "No standard in-text citations (such as `[1]`, `[1, 2]`, or `(Author et al., 2020)`) were detected in the extracted text. "
                    "This may be caused by non-standard citation syntax, footnotes, or complex multi-column layout."
                )
            else:
                st.divider()
                st.subheader(f"🔍 Extracted Citation Sentences ({len(citation_records)} Detected)")
                st.caption("Review extracted sentences before batch classification. You can sort, filter, or classify all at once.")

                df_preview = pd.DataFrame([
                    {
                        "ID": r["id"],
                        "Detected Marker": ", ".join(r["markers"]),
                        "Citation Sentence": r["sentence"],
                        "Words": r["word_count"],
                    }
                    for r in citation_records
                ])
                st.dataframe(df_preview, use_container_width=True, height=220)

                btn_classify_pdf = st.button("🚀 Classify All Extracted Citations", type="primary", key="btn_classify_pdf")

                if btn_classify_pdf or "pdf_classified_results" in st.session_state:
                    if btn_classify_pdf:
                        pdf_model, pdf_tokenizer = load_cached_predictor(pdf_model_path)
                        classified_rows = []
                        prog_bar = st.progress(0.0)
                        status_text = st.empty()

                        t_start = time.perf_counter()
                        for idx, rec in enumerate(citation_records):
                            status_text.text(f"Classifying citation {idx + 1} of {len(citation_records)}...")
                            pred_res = run_single_inference(rec["sentence"], pdf_model, pdf_tokenizer)
                            classified_rows.append({
                                "ID": rec["id"],
                                "Detected Marker": ", ".join(rec["markers"]),
                                "Predicted Intent": pred_res["pred_label"],
                                "Confidence (%)": round(pred_res["confidence"], 2),
                                "Method Prob (%)": round(pred_res["probs"][0] * 100, 2),
                                "Background Prob (%)": round(pred_res["probs"][1] * 100, 2),
                                "Result Prob (%)": round(pred_res["probs"][2] * 100, 2),
                                "Citation Sentence": rec["sentence"],
                                "Latency (ms)": round(pred_res["latency_ms"], 2),
                            })
                            prog_bar.progress((idx + 1) / len(citation_records))

                        t_total = time.perf_counter() - t_start
                        status_text.text(f"✅ Classified {len(citation_records)} citations in {t_total:.2f} seconds!")
                        st.session_state["pdf_classified_results"] = classified_rows

                    results_data = st.session_state.get("pdf_classified_results", [])
                    if results_data:
                        df_res = pd.DataFrame(results_data)

                        st.divider()
                        st.subheader("📊 Paper Citation-Intent Profile")

                        summary_col1, summary_col2 = st.columns([1, 2])

                        with summary_col1:
                            intent_counts = df_res["Predicted Intent"].value_counts().reset_index()
                            intent_counts.columns = ["Intent", "Count"]
                            fig_pie = px.pie(
                                intent_counts,
                                names="Intent",
                                values="Count",
                                title="Paper Citation Intent Distribution",
                                color="Intent",
                                color_discrete_map={"Background": "#3B82F6", "Method": "#10B981", "Result Comparison": "#F59E0B"},
                                hole=0.4,
                            )
                            fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=280)
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with summary_col2:
                            st.markdown("#### Intent Breakdown Summary")
                            for intent_name, color in [("Method", "#10B981"), ("Background", "#3B82F6"), ("Result Comparison", "#F59E0B")]:
                                count = len(df_res[df_res["Predicted Intent"] == intent_name])
                                pct = (count / len(df_res) * 100) if len(df_res) > 0 else 0
                                st.markdown(
                                    f"- <span style='color:{color}; font-weight:600;'>{intent_name}</span>: **{count} citations** ({pct:.1f}% of paper)",
                                    unsafe_allow_html=True,
                                )

                            avg_conf = df_res["Confidence (%)"].mean()
                            st.metric("Average Model Confidence", f"{avg_conf:.2f}%")

                        st.subheader("📋 Detailed Classified Citations Table")
                        st.dataframe(
                            df_res[["ID", "Detected Marker", "Predicted Intent", "Confidence (%)", "Citation Sentence", "Latency (ms)"]],
                            use_container_width=True,
                            height=350,
                        )

                        # CSV Export Button
                        csv_data = df_res.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Classified Citations as CSV",
                            data=csv_data,
                            file_name=f"{Path(uploaded_pdf.name).stem}_citations_classified.csv",
                            mime="text/csv",
                        )
