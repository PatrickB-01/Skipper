"""Train a RandomForest model from labeled session features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RandomForest skip classifier.")
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        required=True,
        help="One or more features CSV paths.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="One or more labels CSV paths (same count/order as --features).",
    )
    parser.add_argument(
        "--positive-labels",
        type=str,
        default="intro,recap,outro,idle",
        help="Comma-separated labels to treat as skip.",
    )
    parser.add_argument("--model-out", type=str, required=True, help="Output model path.")
    parser.add_argument("--feature-out", type=str, required=True, help="Output feature list JSON.")
    parser.add_argument(
        "--error-report",
        type=str,
        default=None,
        help="Optional CSV output for false positives/negatives.",
    )
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


def _load_session_start(features_path: Path) -> float | None:
    session_path = features_path.parent / "session.json"
    if not session_path.exists():
        return None
    try:
        return float(json.loads(session_path.read_text()).get("start_time"))
    except Exception:
        return None


def main() -> None:
    args = _parse_args()
    if len(args.features) != len(args.labels):
        raise ValueError("--features and --labels must have the same number of paths.")

    positive = {name.strip().lower() for name in args.positive_labels.split(",") if name.strip()}

    all_rows: list[dict[str, str]] = []
    all_labels: list[list[dict[str, str]]] = []
    feature_columns: list[str] | None = None

    for features_path, labels_path in zip(args.features, args.labels, strict=False):
        features_file_path = Path(features_path)
        session_start = _load_session_start(features_file_path)
        labels = _load_labels(Path(labels_path))
        with features_file_path.open("r", newline="") as feature_file:
            reader = csv.DictReader(feature_file)
            session_rows = list(reader)

        if not session_rows:
            raise ValueError(f"No feature rows found in {features_path}.")

        current_columns = [
            col for col in session_rows[0].keys() if col not in {"window_id", "timestamp"}
        ]
        if feature_columns is None:
            feature_columns = current_columns
        elif current_columns != feature_columns:
            raise ValueError(
                "Feature columns differ across sessions. Ensure the same feature pipeline is used."
            )

        if session_start is not None:
            for row in session_rows:
                try:
                    row["timestamp"] = str(float(row["timestamp"]) - session_start)
                except Exception:
                    pass

        all_rows.extend(session_rows)
        all_labels.extend([labels] * len(session_rows))

    if not all_rows or feature_columns is None:
        raise ValueError("No feature rows found.")

    X = np.array([[float(row[col]) for col in feature_columns] for row in all_rows], dtype=np.float32)
    y = np.array(
        [
            _label_window(float(row["timestamp"]), labels, positive)
            for row, labels in zip(all_rows, all_labels, strict=False)
        ],
        dtype=np.int32,
    )

    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y if y.sum() > 0 else None,
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

    if args.error_report:
        report_path = Path(args.error_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="") as report_file:
            writer = csv.DictWriter(
                report_file,
                fieldnames=["row_index", "timestamp", "true", "pred", "type"],
            )
            writer.writeheader()
            for idx, true_val, pred_val in zip(idx_test, y_test, y_pred, strict=False):
                if true_val == pred_val:
                    continue
                error_type = "false_negative" if true_val == 1 else "false_positive"
                row = all_rows[int(idx)]
                writer.writerow(
                    {
                        "row_index": int(idx),
                        "timestamp": row.get("timestamp", ""),
                        "true": int(true_val),
                        "pred": int(pred_val),
                        "type": error_type,
                    }
                )
        print(f"Error report saved to {report_path}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        print("\nThreshold report (class=1):")
        print("thr\tprec\trecall\tf1")
        for thr in thresholds:
            pred = (proba >= thr).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test,
                pred,
                average="binary",
                zero_division=0,
            )
            print(f"{thr:.2f}\t{prec:.3f}\t{rec:.3f}\t{f1:.3f}")

    joblib.dump(model, args.model_out)
    Path(args.feature_out).write_text(json.dumps(feature_columns, indent=2))
    print(f"Model saved to {args.model_out}")
    print(f"Feature list saved to {args.feature_out}")


if __name__ == "__main__":
    main()
