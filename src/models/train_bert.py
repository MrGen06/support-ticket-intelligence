import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.models.bert_dataset import TicketDataset

def format_metrics(loss, acc, f1):
    return f"Loss: {loss:.3f} | Acc: {acc*100:.1f}% | Macro F1: {f1*100:.1f}%"

def train_model(sample_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    df = pd.read_csv("data/processed/tickets.csv")
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Running in FAST MODE with {len(df)} samples.")

    # 2. Encode Labels
    labels = df['queue'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df['label'] = df['queue'].map(label2id)

    # Save mapping as required by PDF
    os.makedirs("models/distilbert", exist_ok=True)
    with open("models/distilbert/queue_label_mapping.json", "w") as f:
        json.dump(label2id, f, indent=4)

    # 3. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 4. Tokenizer & Datasets
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = TicketDataset(X_train, y_train, tokenizer, max_length=256)
    val_dataset = TicketDataset(X_val, y_val, tokenizer, max_length=256)
    test_dataset = TicketDataset(X_test, y_test, tokenizer, max_length=256)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 5. Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 2

    # 6. Training Loop (matching PDF output format)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Train
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(batch_labels.cpu().numpy())

        t_acc = accuracy_score(train_labels, train_preds)
        t_f1 = f1_score(train_labels, train_preds, average='macro')
        print(f"Train {format_metrics(train_loss/len(train_loader), t_acc, t_f1)}")

        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_labels.cpu().numpy())

        v_acc = accuracy_score(val_labels, val_preds)
        v_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Validation {format_metrics(val_loss/len(val_loader), v_acc, v_f1)}")

    # 7. Final Test Evaluation
    print("\nFINAL TEST")
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(batch_labels.cpu().numpy())
            
    test_acc = accuracy_score(test_labels, test_preds)
    test_macro = f1_score(test_labels, test_preds, average='macro')
    test_weighted = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"Accuracy: {test_acc*100:.1f}%")
    print(f"Macro F1: {test_macro*100:.1f}%")
    print(f"Weighted F1: {test_weighted*100:.1f}%\n")

    # 8. Export Artifacts exactly as PDF requires
    model.save_pretrained("models/distilbert", safe_serialization=True)
    tokenizer.save_pretrained("models/distilbert")
    print("Exported model and tokenizer artifacts to models/distilbert/")

if __name__ == "__main__":
    import sys
    # If run with 'python src/models/train_bert.py --fast', it uses 50 rows.
    sample = 50 if '--fast' in sys.argv else None
    train_model(sample_size=sample)