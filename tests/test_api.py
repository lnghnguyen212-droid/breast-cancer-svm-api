from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_rejects_missing_features():
    r = client.post("/predict", json={"features": [10.0]})
    assert r.status_code == 400