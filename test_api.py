from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_returns_required_keys():
    r = client.post("/predict", json={"text": "my computer is broken, urgent, help immediately"})
    assert r.status_code == 200
    body = r.json()
    assert {"category", "confidence", "urgency"}.issubset(body.keys())


def test_predict_rejects_bad_input():
    r = client.post("/predict", json={"wrong_field": "hi"})
    assert r.status_code == 422   # pydantic bouncer kicked in


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "f1" in r.json()