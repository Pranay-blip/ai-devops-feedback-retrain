import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="number of feedback rows")
    parser.add_argument("--bad", action="store_true", help="use wrong labels")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    X, y = load_iris(return_X_y=True)
    idx = rng.choice(len(X), size=args.n, replace=False)

    rows = []
    for i in idx:
        values = (X[i] + rng.normal(0, 0.1, 4)).round(2).tolist()
        label = int(y[i])
        if args.bad:
            label = (label + 1) % 3  # deliberately wrong
        rows.append(values + [label])

    path = Path("data/feedback.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLS + ["label"])
        writer.writerows(rows)
    kind = "BAD" if args.bad else "good"
    print(f"Wrote {len(rows)} {kind} feedback rows to {path}")


if __name__ == "__main__":
    main()