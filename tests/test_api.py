from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_trips():
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "trips" in data
    assert isinstance(data["trips"], list)


def test_docs():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/trips" in paths

def test_predict_unknown_trip():
    response = client.post("/predict/nonexistent_trip")
    assert response.status_code == 404


def test_predict_pipeline_unavailable():
    trips = client.get("/trips").json()["trips"]

    if not trips:
        return

    response = client.post(f"/predict/{trips[0]}")
    assert response.status_code == 503
    assert response.json()["detail"]["message"] == "Prediction pipeline not ready"
