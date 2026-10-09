from fastapi.testclient import TestClient

import src.app as app_module

client = TestClient(app_module.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_setosa():
    r = client.post("/predict", json={"values": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.json() == {"prediction": 0}


def test_predict_rejects_wrong_number_of_values():
    r = client.post("/predict", json={"values": [1.0, 2.0, 3.0]})
    assert r.status_code == 422


def test_feedback_is_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FEEDBACK_PATH", tmp_path / "fb.csv")
    r = client.post("/feedback", json={"values": [6.3, 3.3, 6.0, 2.5], "label": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["stored"] is True
    assert body["feedback_count"] == 1
    assert body["retrain_needed"] is False


def test_feedback_rejects_invalid_label(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FEEDBACK_PATH", tmp_path / "fb.csv")
    r = client.post("/feedback", json={"values": [6.3, 3.3, 6.0, 2.5], "label": 5})
    assert r.status_code == 422