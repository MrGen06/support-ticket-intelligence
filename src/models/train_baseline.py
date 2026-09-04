import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

def main():
    data_path = "data/processed/tickets.csv"
    
    # Fallback to mock data if the user hasn't placed their real dataset yet
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating mock data for pipeline verification...")
        df = pd.DataFrame({
            "text": [
                "Unable to access the dashboard", "Billing page is crashing", 
                "How do I reset my password?", "Server is down", 
                "Payment failed", "Feature request for dark mode",
                "Cannot login to my account", "Invoice is incorrect",
                "App is crashing on startup", "Where are the settings?"
            ] * 10,  # Multiplied to create enough samples for splitting
            "queue": [
                "Technical Support", "Billing", 
                "Customer Service", "Technical Support", 
                "Billing", "General Inquiry",
                "Technical Support", "Billing",
                "Technical Support", "General Inquiry"
            ] * 10,
            "priority": [
                "High", "Medium", 
                "Low", "Critical", 
                "High", "Low",
                "High", "Medium",
                "Critical", "Low"
            ] * 10
        })
    else:
        df = pd.read_csv(data_path)

    print("Training baseline models...")
    
    # Split data
    X = df["text"]
    y_queue = df["queue"]
    y_priority = df["priority"]
    
    X_train, X_test, y_q_train, y_q_test, y_p_train, y_p_test = train_test_split(
        X, y_queue, y_priority, test_size=0.2, random_state=42
    )
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Queue Model
    queue_clf = LogisticRegression(max_iter=1000)
    queue_clf.fit(X_train_vec, y_q_train)
    
    # Train Priority Model
    priority_clf = LogisticRegression(max_iter=1000)
    priority_clf.fit(X_train_vec, y_p_train)
    
    # Evaluate Queue
    q_preds = queue_clf.predict(X_test_vec)
    q_acc = accuracy_score(y_q_test, q_preds)
    q_f1 = f1_score(y_q_test, q_preds, average="macro")
    
    # Evaluate Priority
    p_preds = priority_clf.predict(X_test_vec)
    p_acc = accuracy_score(y_p_test, p_preds)
    p_f1 = f1_score(y_p_test, p_preds, average="macro")
    
    print("\n--- Baseline Metrics ---")
    print(f"Queue Accuracy: {q_acc:.4f} | Queue Macro F1: {q_f1:.4f}")
    print(f"Priority Accuracy: {p_acc:.4f} | Priority Macro F1: {p_f1:.4f}")
    
    # Save artifacts
    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(vectorizer, "models/baseline/tfidf_vectorizer.joblib")
    joblib.dump(queue_clf, "models/baseline/queue_classifier.joblib")
    joblib.dump(priority_clf, "models/baseline/priority_classifier.joblib")
    print("\nBaseline models saved to models/baseline/")

if __name__ == "__main__":
    main()