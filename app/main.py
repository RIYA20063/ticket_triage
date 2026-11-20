import json
from fastapi import FastAPI
from pydantic import BaseModel
from src.model import TicketClassifier
from src.database import init_db, log_prediction, get_history

app = FastAPI(title="Ticket Triage API")

# Runs once when the server starts
init_db()
classifier = TicketClassifier()


class Ticket(BaseModel):
    """Defines what a valid request looks like: just a 'text' field."""
    text: str


@app.post("/predict")
def predict(ticket: Ticket):
    """Receive ticket text -> return category, confidence, urgency. Log it to SQL."""
    result = classifier.predict(ticket.text)
    log_prediction(ticket.text, result["category"],
                   result["urgency"], result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    """Return the model's evaluation scores."""
    with open("metrics.json") as f:
        return json.load(f)


@app.get("/history")
def history():
    """Return the last 20 predictions from the database."""
    return [
        {"id": r[0], "text": r[1], "category": r[2],
         "urgency": r[3], "confidence": r[4], "time": r[5]}
        for r in get_history()
    ]
@app.get("/")
def root():
    return {"message": "Ticket Triage API is running. See /docs"}