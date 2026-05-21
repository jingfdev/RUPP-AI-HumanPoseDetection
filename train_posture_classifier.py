"""Train and evaluate Normal/Sleeping posture classifiers from extracted pose features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DROP_COLUMNS = {"image_path", "label", "mapped_label", "model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train posture classifiers from YOLO pose features.")
    parser.add_argument("--features", default="posture_dataset/features/posture_features.csv", help="Feature CSV path.")
    parser.add_argument("--output-dir", default="posture_models", help="Output directory for models and reports.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column not in DROP_COLUMNS]


def candidate_models(random_state: int) -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def save_confusion_matrix(y_true, y_pred, labels: List[str], path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df = df[df["mapped_label"].isin(["normal", "sleeping"])].copy()
    if df["mapped_label"].nunique() < 2:
        raise RuntimeError("Need at least two classes: normal and sleeping.")
    if len(df) < 10:
        raise RuntimeError("Need more feature rows before training. Collect more dataset frames first.")

    columns = feature_columns(df)
    x = df[columns].astype(float)
    y = df["mapped_label"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    metrics_rows = []
    best_name = ""
    best_model = None
    best_f1 = -1.0
    labels = ["normal", "sleeping"]

    for name, model in candidate_models(args.random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
        }
        metrics_rows.append(metrics)

        report = classification_report(y_test, predictions, labels=labels, zero_division=0)
        (output_dir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")
        save_confusion_matrix(
            y_test,
            predictions,
            labels,
            output_dir / f"{name}_confusion_matrix.png",
            f"{name} Confusion Matrix",
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_name = name
            best_model = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(output_dir / "classifier_metrics.csv", index=False)
    assert best_model is not None
    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "feature_columns": columns,
            "labels": labels,
        },
        output_dir / "best_posture_classifier.joblib",
    )

    summary = {
        "best_model": best_name,
        "best_f1_macro": best_f1,
        "feature_columns": columns,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "class_counts": y.value_counts().to_dict(),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"[INFO] Best model: {best_name}")
    print(f"[INFO] Saved: {output_dir / 'best_posture_classifier.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
