"""Research Report and Export Center Page."""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Research Report & Exports", page_icon="📑", layout="wide")

st.title("📑 Academic Research Report & Data Export Center")
st.markdown("Review the formal research report synthesized from experimental measurements and download raw artifacts.")

results_dir = Path("results")
report_path = results_dir / "research_report.md"

# Action buttons for exports
st.subheader("📥 Export Experimental Artifacts")
e_col1, e_col2, e_col3, e_col4 = st.columns(4)

with e_col1:
    bench_csv = results_dir / "benchmark_table.csv"
    if bench_csv.exists():
        with open(bench_csv, "rb") as f:
            st.download_button("📊 Download Benchmarks CSV", f, file_name="kd_citation_benchmarks.csv", mime="text/csv")
    else:
        st.button("📊 Benchmarks CSV (Pending)", disabled=True)

with e_col2:
    rob_csv = results_dir / "robustness_table.csv"
    if rob_csv.exists():
        with open(rob_csv, "rb") as f:
            st.download_button("🛡️ Download Robustness CSV", f, file_name="kd_citation_robustness.csv", mime="text/csv")
    else:
        st.button("🛡️ Robustness CSV (Pending)", disabled=True)

with e_col3:
    manifest_json = results_dir / "manifest.json"
    if manifest_json.exists():
        with open(manifest_json, "rb") as f:
            st.download_button("📋 Download Manifest JSON", f, file_name="experiment_manifest.json", mime="application/json")
    else:
        st.button("📋 Manifest JSON (Pending)", disabled=True)

with e_col4:
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button("📝 Download Full Report (MD)", f, file_name="research_report.md", mime="text/markdown")
    else:
        st.button("📝 Full Report (Pending)", disabled=True)

st.divider()

# Render Report
if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)
else:
    st.info("The research report has not been generated yet. Run `python scripts/generate_report.py` to compile all empirical data.")
