import os
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from html import escape

st.set_page_config(
    page_title="Support Ticket AI",
    layout="wide"
)

# Initialize session state to hold ticket history dynamically
if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []

st.title("AI Support Ticket Intelligence")

st.markdown(
    """
    <style>
    .ticket-result {
        min-height: 6.5rem;
        padding: 0.85rem 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.5rem;
    }
    .ticket-result-label {
        color: rgba(128, 128, 128, 0.95);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .ticket-result-value {
        margin-top: 0.45rem;
        font-size: clamp(1rem, 2vw, 1.45rem);
        font-weight: 650;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .ticket-result-detail {
        margin-top: 0.35rem;
        color: rgba(128, 128, 128, 0.95);
        font-size: 0.85rem;
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

                # Save dynamic metrics to session state
                st.session_state.ticket_history.append({
                    "Queue": data["queue"],
                    "Priority": data["priority"],
                    "Confidence": data["confidence"],
                    "Latency (ms)": data["ml_latency_ms"]
                })

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div class="ticket-result"><div class="ticket-result-label">Queue</div>'
                        f'<div class="ticket-result-value">{escape(str(data["queue"]))}</div></div>',
                        unsafe_allow_html=True,
                    )
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
                    f"{deadline.strftime('%I:%M %p').lstrip('0')} UTC"
                )
                duration = (
                    f"SLA window: {sla_info['sla_hours']} hours"
                    if sla_info.get("sla_hours") is not None
                    else ""
                )
                st.markdown(
                    f'<div class="ticket-result"><div class="ticket-result-label">Respond by</div>'
                    f'<div class="ticket-result-value">{escape(deadline_text)}</div>'
                    f'<div class="ticket-result-detail">{escape(duration)}</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("View exact UTC timestamp"):
                    st.code(sla_info["deadline"])

                st.subheader("AI Summary & Response")
                st.write(data["response_draft"])

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")

# ==========================================
# Dynamic Live Dashboard
# ==========================================
st.divider()
st.subheader("Live System Analytics")

if len(st.session_state.ticket_history) > 0:
    df = pd.DataFrame(st.session_state.ticket_history)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ticket Volume by Queue**")
        queue_counts = df["Queue"].value_counts().reset_index()
        queue_counts.columns = ["Queue", "Count"]
        st.bar_chart(queue_counts.set_index("Queue"))

    with col2:
        st.markdown("**Model Inference Latency (ms)**")
        # Line chart tracks latency spikes over time
        st.line_chart(df["Latency (ms)"])

    st.caption(
        f"**System Performance:** Processed {len(df)} tickets | "
        f"Avg Confidence: {df['Confidence'].mean():.1%} | "
        f"Avg Latency: {df['Latency (ms)'].mean():.0f} ms"
    )
else:
    st.info("Process tickets above to generate dynamic live charts and model performance metrics.")