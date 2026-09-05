import os
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Support Ticket AI",
    layout="wide"
)

st.title("AI Support Ticket Intelligence")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: clamp(1rem, 2.2vw, 1.75rem);
        line-height: 1.2;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

subject = st.text_input("Ticket Subject")
body = st.text_area("Ticket Description", height=180)

# Use env variable for Docker compatibility, fallback to localhost for local testing
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

if st.button("Analyze Ticket"):
    if not subject or not body:
        st.warning("Please enter both a subject and a description.")
    else:
        with st.spinner("Analyzing ticket..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"subject": subject, "body": body}
                )
                response.raise_for_status()
                data = response.json()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Queue", data["queue"])
                c2.metric("Priority", data["priority"])
                c3.metric("ML Latency", f"{data['ml_latency_ms']:.0f} ms")
                
                st.subheader("SLA Deadline")
                sla_info = data["sla"] if isinstance(data["sla"], dict) else {
                    "deadline": data["sla"],
                    "sla_hours": None,
                }
                deadline = datetime.fromisoformat(sla_info["deadline"])
                deadline_text = (
                    f"{deadline.strftime('%b')} {deadline.day}, {deadline.year} at "
                    f"{deadline.strftime('%I:%M:%S %p').lstrip('0')} UTC"
                )
                deadline_col, duration_col = st.columns(2)
                deadline_col.metric("Respond by", deadline_text)
                if sla_info["sla_hours"] is not None:
                    duration_col.metric("SLA window", f"{sla_info['sla_hours']} hours")
                with st.expander("View exact timestamp"):
                    st.code(sla_info["deadline"])
                
                st.subheader("AI Summary & Response")
                st.write(data["response_draft"])
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")

st.divider()
st.subheader("Ticket Trends")
trend_data = pd.DataFrame({
    "Queue": ["Technical", "Product", "Billing", "Customer"],
    "Tickets": [42, 31, 18, 27]
})
st.bar_chart(trend_data.set_index("Queue"))