import time
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.priority import PriorityPredictor

class TicketInference:
    def __init__(self, bert_dir="models/distilbert", baseline_dir="models/baseline"):
        print("Loading ML models into memory...")
        self.device = torch.device("cpu")
        
        # 1. Load DistilBERT Queue Model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert_queue = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        self.bert_queue.to(self.device)
        self.bert_queue.eval()
        
        # 2. Load Scikit-Learn Baseline (Queue)
        self.tfidf = joblib.load(f"{baseline_dir}/tfidf_vectorizer.joblib")
        self.baseline_queue = joblib.load(f"{baseline_dir}/queue_classifier.joblib")
        
        # 3. Load Priority Model (via our wrapper)
        self.priority_predictor = PriorityPredictor(
            f"{baseline_dir}/tfidf_vectorizer.joblib",
            f"{baseline_dir}/priority_classifier.joblib"
        )
        print("Models loaded successfully.")

    def predict(self, text):
        start = time.perf_counter()
        
        # A. DistilBERT Queue Inference
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        
        with torch.inference_mode():
            outputs = self.bert_queue(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, idx = torch.max(probs, dim=-1)
            
        bert_latency = (time.perf_counter() - start) * 1000
        
        # B. Baseline Queue Inference (for fallback/comparison)
        baseline_prediction = self.baseline_queue.predict(self.tfidf.transform([text]))[0]
        
        # C. Priority Inference
        priority_prediction = self.priority_predictor.predict(text)
        
        return {
            "queue": self.bert_queue.config.id2label[idx.item()],
            "confidence": round(confidence.item(), 4),
            "baseline_queue": baseline_prediction,
            "priority": priority_prediction,
            "latency_ms": round(bert_latency, 2)
        }