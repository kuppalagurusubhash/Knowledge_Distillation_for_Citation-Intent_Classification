"""Dataset Explorer Page: SciCite analysis, split distributions, and sample exploration."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datasets import load_dataset

st.set_page_config(page_title="Dataset Explorer - SciCite", page_icon="📊", layout="wide")

st.title("📊 SciCite Dataset Explorer")
st.markdown("Detailed exploration of the official **SciCite** citation-intent dataset, class imbalance, and split statistics.")

summary_path = Path("results/dataset_summary.json")

if not summary_path.exists():
    st.warning("Dataset summary not found. Please run `python scripts/download_data.py` first.")
    st.stop()

with open(summary_path, "r", encoding="utf-8") as f:
    stats = json.load(f)

splits = stats.get("splits", {})

# Top split cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Train Split", f"{splits.get('train', {}).get('total_samples', 0):,} citations")
with c2:
    st.metric("Validation Split", f"{splits.get('validation', {}).get('total_samples', 0):,} citations")
with c3:
    st.metric("Test Split (Holdout)", f"{splits.get('test', {}).get('total_samples', 0):,} citations")
with c4:
    leak_check = stats.get("data_leakage_check", {})
    overlap = leak_check.get("train_test_overlap_count", 0)
    st.metric("Train-Test Leakage", f"{overlap} overlaps", delta="Clean" if overlap == 0 else "Warning")

st.divider()

# Class Distribution Section
st.subheader("⚖️ Intent Class Distribution & Imbalance")

dist_data = []
for split_name, split_info in splits.items():
    dist = split_info.get("label_distribution", {})
    for label, vals in dist.items():
        dist_data.append({
            "Split": split_name.capitalize(),
            "Citation Intent": label.capitalize(),
            "Count": vals.get("count", 0),
            "Percentage (%)": vals.get("percentage", 0.0),
        })

df_dist = pd.DataFrame(dist_data)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_bar = px.bar(
        df_dist,
        x="Split",
        y="Count",
        color="Citation Intent",
        barmode="group",
        title="Sample Counts per Intent by Split",
        color_discrete_map={"Background": "#3B82F6", "Method": "#10B981", "Result": "#F59E0B"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_chart2:
    train_dist = splits.get("train", {}).get("label_distribution", {})
    fig_pie = px.pie(
        values=[v["count"] for v in train_dist.values()],
        names=[k.capitalize() for k in train_dist.keys()],
        title="Training Set Class Proportion (Severe Imbalance on 'Result')",
        color=[k.capitalize() for k in train_dist.keys()],
        color_discrete_map={"Background": "#3B82F6", "Method": "#10B981", "Result": "#F59E0B"},
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.info(
    "💡 **Imbalance Insight**: The `background` intent dominates at ~59%, while the critical `result` comparison intent "
    "represents only ~13-14% of citations. This underscores the necessity of macro-F1 and per-class minority evaluation."
)

st.divider()

# Word Length Distribution
st.subheader("📏 Citation Sentence Length Statistics")
col_len1, col_len2, col_len3, col_len4 = st.columns(4)
train_len = splits.get("train", {}).get("word_length", {})

with col_len1:
    st.metric("Mean Word Count", f"{train_len.get('mean', 0):.1f} words")
with col_len2:
    st.metric("Median Word Count", f"{train_len.get('median', 0):.1f} words")
with col_len3:
    st.metric("95th Percentile", f"{train_len.get('p95', 0):.1f} words")
with col_len4:
    st.metric("Max Word Count", f"{train_len.get('max', 0)} words")

st.divider()

# Interactive Sample Browser
st.subheader("🔍 Citation Sentence Browser")
selected_intent = st.selectbox("Filter citations by Intent Class:", ["All", "Background", "Method", "Result"])

@st.cache_data
def get_sample_citations():
    try:
        ds = load_dataset("scicite", trust_remote_code=True)
        train_df = ds["train"].to_pandas().head(200)
        id2label = {0: "Method", 1: "Background", 2: "Result"}
        train_df["Intent"] = train_df["label"].map(id2label)
        return train_df[["string", "Intent", "sectionName", "id"]]
    except Exception:
        return pd.DataFrame()

df_samples = get_sample_citations()
if not df_samples.empty:
    if selected_intent != "All":
        filtered_df = df_samples[df_samples["Intent"] == selected_intent]
    else:
        filtered_df = df_samples

    st.dataframe(
        filtered_df.rename(columns={"string": "Citation Sentence", "sectionName": "Section"}),
        use_container_width=True,
        height=350,
    )
