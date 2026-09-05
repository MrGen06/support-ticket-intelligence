import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Support Ticket AI",
    layout="wide"
)

st.title("AI Support Ticket Intelligence")

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
                # Handle dictionary format returned by our FastAPI SLA engine
                deadline = data["sla"]["deadline"] if isinstance(data["sla"], dict) else data["sla"]
                st.write(deadline)
                
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