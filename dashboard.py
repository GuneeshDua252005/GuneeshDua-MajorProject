from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from emotion_adaptive_system import EmotionalAnalytics


st.set_page_config(page_title="Emotion Intelligence Dashboard", layout="wide")
st.title("Cognitive Emotion Intelligence & Adaptive Lifestyle Dashboard")

storage_path = st.sidebar.text_input("Excel data file", value="emotion_logs.xlsx")
file_path = Path(storage_path)

if pd is None:
    st.error("pandas is required for dashboard analytics.")
    st.stop()

if not file_path.exists():
    st.warning("No data file found yet. Run the system once to create logs.")
    st.stop()

try:
    df = pd.read_excel(file_path, sheet_name="interactions")
except Exception as exc:
    st.error(f"Failed to read Excel data: {exc}")
    st.stop()

analytics = EmotionalAnalytics()
summary = analytics.build_summary(df)

col1, col2, col3 = st.columns(3)
col1.metric("Dominant Emotion", summary.get("dominant_emotion", "n/a"))
col2.metric("Average Emotional Energy", summary.get("average_energy_score", 0))
col3.metric("Anomaly Count", len(summary.get("anomalies", [])))

st.subheader("Mood Distribution")
distribution = summary.get("mood_distribution", {})
if distribution:
    dist_df = pd.DataFrame(
        [{"emotion": emotion, "ratio": ratio} for emotion, ratio in distribution.items()]
    ).sort_values("ratio", ascending=False)
    st.bar_chart(dist_df.set_index("emotion"))
else:
    st.info("No distribution available yet.")

st.subheader("Engagement by Day")
engagement = summary.get("engagement_by_day", {})
if engagement:
    engagement_df = pd.DataFrame(
        [{"date": d, "interactions": c} for d, c in engagement.items()]
    ).sort_values("date")
    st.line_chart(engagement_df.set_index("date"))
else:
    st.info("No engagement trend available yet.")

st.subheader("Detected Anomalies")
anomalies = summary.get("anomalies", [])
if anomalies:
    st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
else:
    st.success("No concerning anomalies found in current data window.")
