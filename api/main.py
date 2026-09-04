from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import TicketInference
from src.llm_assistant import LLMAssistant
from src.sla import calculate_sla

app = FastAPI(title="AI Support Ticket Assistant")

# Initialize and load everything into memory at startup
classifier = TicketInference(
    bert_dir="models/distilbert",
    baseline_dir="models/baseline"
)
llm = LLMAssistant()

class TicketRequest(BaseModel):
    subject: str
    body: str

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "distilbert"
    }

@app.post("/predict")
def predict(ticket: TicketRequest):
    text = ticket.subject + "\n" + ticket.body
    
    # 1. Run ML Inference (Queue + Priority)
    prediction = classifier.predict(text)
    priority = prediction["priority"]
    
    # 2. Calculate SLA
    sla_info = calculate_sla(priority)
    
    # 3. Generate LLM Draft using Qwen/Groq
    generated = llm.assist(
        ticket=text,
        queue=prediction["queue"],
        priority=priority,
        sla=sla_info["deadline"]
    )
    
    # 4. Return everything needed by the dashboard
    return {
        "queue": prediction["queue"],
        "confidence": prediction["confidence"],
        "priority": priority,
        "sla": sla_info,
        "response_draft": generated,
        "ml_latency_ms": prediction["latency_ms"]
    }