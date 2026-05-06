"""Train a RandomForest model from labeled session features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RandomForest skip classifier.")
    parser.add_argument("--features", type=str, required=True, help="Features CSV path.")
    parser.add_argument("--labels", type=str, required=True, help="Labels CSV path.")
    parser.add_argument(
        "--positive-labels",
        type=str,
        default="intro,recap,outro",
        help="Comma-separated labels to treat as skip.",
    )
    parser.add_argument("--model-out", type=str, required=True, help="Output model path.")
    parser.add_argument("--feature-out", type=str, required=True, help="Output feature list JSON.")
    return parser.parse_args()


def _load_labels(label_path: Path) -> list[dict[str, str]]:
    with label_path.open("r", newline="") as label_file:
        reader = csv.DictReader(label_file)
        return list(reader)


def _label_window(timestamp: float, labels: list[dict[str, str]], positive: set[str]) -> int:
    for label in labels:
        start = float(label["start"])
        end = float(label["end"])
        name = label["label"].strip().lower()
        if start <= timestamp <= end and name in positive:
            return 1
    return 0


def main() -> None:
    args = _parse_args()
    feature_path = Path(args.features)
    label_path = Path(args.labels)

    positive = {name.strip().lower() for name in args.positive_labels.split(",") if name.strip()}
    labels = _load_labels(label_path)

    with feature_path.open("r", newline="") as feature_file:
        reader = csv.DictReader(feature_file)
        rows = list(reader)

    if not rows:
        raise ValueError("No feature rows found.")

    feature_columns = [col for col in rows[0].keys() if col not in {"window_id", "timestamp"}]

    X = np.array([[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float32)
    y = np.array([
        _label_window(float(row["timestamp"]), labels, positive) for row in rows
    ], dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 0 else None
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(model, args.model_out)
    Path(args.feature_out).write_text(json.dumps(feature_columns, indent=2))
    print(f"Model saved to {args.model_out}")
    print(f"Feature list saved to {args.feature_out}")


if __name__ == "__main__":
    main()
