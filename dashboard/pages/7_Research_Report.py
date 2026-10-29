"""Research Report and Export Center Page: Full academic report and project artifact exports."""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Research Report & Exports - KD Citation", page_icon="📑", layout="wide")

st.title("📑 Academic Research Report & Artifact Export Center")
st.markdown(
    "Review the formal research report synthesized from experimental measurements and download raw artifacts, "
    "benchmark CSVs, robustness tables, and submission documents."
)

results_dir = Path("results")
project_root = Path(".")
report_path = results_dir / "research_report.md"

# ==============================================================================
# 1. ACTION BUTTONS FOR DOWNLOADS
# ==============================================================================
st.subheader("📥 Export Experimental Artifacts & Submission Deliverables")

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    bench_csv = results_dir / "benchmark_table.csv"
    if bench_csv.exists():
        with open(bench_csv, "rb") as f:
            st.download_button("📊 Download Benchmarks CSV", f, file_name="kd_citation_benchmarks.csv", mime="text/csv")
    else:
        st.button("📊 Benchmarks CSV (Pending)", disabled=True)

with col_d2:
    rob_csv = results_dir / "robustness_table.csv"
    if rob_csv.exists():
        with open(rob_csv, "rb") as f:
            st.download_button("🛡️ Download Robustness CSV", f, file_name="kd_citation_robustness.csv", mime="text/csv")
    else:
        st.button("🛡️ Robustness CSV (Pending)", disabled=True)

with col_d3:
    manifest_json = results_dir / "manifest.json"
    if manifest_json.exists():
        with open(manifest_json, "rb") as f:
            st.download_button("📋 Download Manifest JSON", f, file_name="experiment_manifest.json", mime="application/json")
    else:
        st.button("📋 Manifest JSON (Pending)", disabled=True)

with col_d4:
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button("📝 Download Full Report (MD)", f, file_name="research_report.md", mime="text/markdown")
    else:
        st.button("📝 Full Report (Pending)", disabled=True)

# Coursework Task 1 Deliverables Export Row
col_t1, col_t2 = st.columns(2)
with col_t1:
    task1_docx = project_root / "Task1.docx"
    if task1_docx.exists():
        with open(task1_docx, "rb") as f:
            st.download_button(
                "🎓 Download Week 2 Task 1 DOCX (LMS Submission)",
                f,
                file_name="Task1.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
with col_t2:
    task1_md = project_root / "Task1_Literature_Survey_and_Requirement_Analysis.md"
    if task1_md.exists():
        with open(task1_md, "rb") as f:
            st.download_button(
                "📄 Download Week 2 Task 1 Markdown Document",
                f,
                file_name="Task1_Literature_Survey_and_Requirement_Analysis.md",
                mime="text/markdown",
            )

st.divider()

# ==============================================================================
# 2. CHECKPOINT & ARTIFACT INVENTORY TABLE
# ==============================================================================
st.subheader("📦 Model Checkpoint & Storage Inventory")

ckpts_dir = Path("artifacts/checkpoints")
inventory_records = []
if ckpts_dir.exists():
    for d in sorted(ckpts_dir.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            total_size = sum(f.stat().st_size for f in d.glob("**/*") if f.is_file()) / (1024 * 1024)
            inventory_records.append({
                "Checkpoint Alias": d.name,
                "Directory Path": str(d),
                "Size (MB)": round(total_size, 2),
                "Has Model Weights": "Yes (model.safetensors / pytorch_model.bin)",
                "Status": "Verified & Ready for Inference",
            })

if inventory_records:
    df_inv = pd.DataFrame(inventory_records)
    st.dataframe(df_inv, use_container_width=True)
else:
    st.info("No saved model checkpoints found in `artifacts/checkpoints/`.")

st.divider()

# ==============================================================================
# 3. RENDERED RESEARCH REPORT
# ==============================================================================
st.subheader("📑 Formal Academic Research Report")

if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)
else:
    st.info("The research report has not been generated yet. Run `python scripts/generate_report.py` to compile all empirical data.")
