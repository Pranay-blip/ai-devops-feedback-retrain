import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEEDBACK = Path("data/feedback.csv")
MODELS = Path("models")
THRESHOLD = 0.90


def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    X.columns = COLS
    y = iris.target.copy()
    return X, y


def train_candidate():
    X, y = load_data()
    # fixed validation split from the ORIGINAL data, so models are comparable
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_feedback = 0
    if FEEDBACK.exists():
        fb = pd.read_csv(FEEDBACK)
        n_feedback = len(fb)
        X_train = pd.concat([X_train, fb[COLS]], ignore_index=True)
        y_train = pd.concat([y_train, fb["label"]], ignore_index=True)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "candidate.joblib")
    meta = {
        "accuracy": round(acc, 4),
        "train_samples": int(len(X_train)),
        "feedback_samples": int(n_feedback),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (MODELS / "candidate_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"candidate_accuracy={acc:.4f} meta={meta}")

    if acc < THRESHOLD:
        sys.exit("Quality gate failed: accuracy below threshold")
    return acc


if __name__ == "__main__":
    train_candidate()