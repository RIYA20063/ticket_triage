import joblib
from src.preprocess import clean_text

# Make the raw dataset labels sound like a triage system

# categories in the SAME order as CATEGORIES in train.py
# model's targets are integers 0,1,2,3
TARGET_NAMES = ['sports', 'tech', 'science', 'politics']
# v1 rule: urgency from keywords (be honest about this in interviews!)
URGENT_WORDS = {"urgent", "immediately", "asap", "emergency", "critical", "broken", "help"}


class TicketClassifier:
    """Loads the trained model and predicts category + urgency for one ticket."""

    def __init__(self, model_path="model.joblib"):
        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)

        probs = self.pipeline.predict_proba([cleaned])[0]   # probability for each category
        idx = int(probs.argmax())  # most likely one
        confidence = float(probs.max())                     # how sure is it?

        return {
            "category": TARGET_NAMES[idx],
            "confidence": round(confidence, 3),
            "urgency": self._urgency(cleaned),
        }

    def _urgency(self, text: str) -> str:
        hits = sum(1 for w in URGENT_WORDS if w in text.lower())
        return "high" if hits >= 2 else "medium" if hits == 1 else "low"