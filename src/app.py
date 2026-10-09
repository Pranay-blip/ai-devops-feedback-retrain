import csv
import json
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/model_meta.json")
FEEDBACK_PATH = Path("data/feedback.csv")
COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
RETRAIN_THRESHOLD = 20

app = FastAPI(title="Iris Feedback API")
model = joblib.load(MODEL_PATH)


class Features(BaseModel):
    values: list[float]


class Feedback(BaseModel):
    values: list[float]
    label: int


def check_values(values):
    if len(values) != 4:
        raise HTTPException(status_code=422, detail="Exactly 4 feature values required")


def feedback_count():
    if not FEEDBACK_PATH.exists():
        return 0
    with FEEDBACK_PATH.open() as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


@app.post("/predict")
def predict(item: Features):
    check_values(item.values)
    return {"prediction": int(model.predict([item.values])[0])}


@app.post("/feedback")
def feedback(item: Feedback):
    check_values(item.values)
    if item.label not in (0, 1, 2):
        raise HTTPException(status_code=422, detail="label must be 0, 1 or 2")
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLS + ["label"])
        writer.writerow(item.values + [item.label])
    count = feedback_count()
    return {
        "stored": True,
        "feedback_count": count,
        "retrain_needed": count >= RETRAIN_THRESHOLD,
    }


@app.get("/feedback/count")
def count():
    n = feedback_count()
    return {
        "feedback_count": n,
        "threshold": RETRAIN_THRESHOLD,
        "retrain_needed": n >= RETRAIN_THRESHOLD,
    }