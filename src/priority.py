import joblib

class PriorityPredictor:
    def __init__(self, vectorizer_path, model_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

    def predict(self, text):
        vec_text = self.vectorizer.transform([text])
        return self.model.predict(vec_text)[0]