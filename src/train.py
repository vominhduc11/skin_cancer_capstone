from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    ARTIFACT_DIR,
    CSV_DEFAULT_PATH,
    MODEL_FEATURES,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data_utils import (
    add_diagnosis_column,
    build_binary_dataset,
    build_multiclass_dataset,
    build_svm_model,
    evaluate_predictions,
    get_feature_matrix,
    load_metadata,
    save_artifact,
    save_confusion_matrix,
    save_text_report,
)


def train_binary(df):
    df_binary = build_binary_dataset(df)

    X = get_feature_matrix(df_binary)
    y = df_binary["target_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_svm_model(probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    class_names = ["benign_group", "high_risk_group"]
    metrics = evaluate_predictions(y_test, y_pred, class_names)

    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        "Binary Classification - Confusion Matrix",
        OUTPUT_DIR / "binary_confusion_matrix.png",
    )
    save_text_report(
        metrics["classification_report_text"],
        OUTPUT_DIR / "binary_classification_report.txt",
    )
    save_artifact(
        ARTIFACT_DIR / "binary_svm.joblib",
        model=model,
        task_name="binary",
        feature_columns=MODEL_FEATURES,
        class_names=class_names,
    )
    return metrics


def train_multiclass(df):
    df_multi = build_multiclass_dataset(df)

    X = get_feature_matrix(df_multi)
    y_raw = df_multi["dx"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_svm_model(probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    class_names = list(label_encoder.classes_)

    metrics = evaluate_predictions(y_test, y_pred, class_names)

    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        "Multi-class Classification - Confusion Matrix",
        OUTPUT_DIR / "multiclass_confusion_matrix.png",
    )
    save_text_report(
        metrics["classification_report_text"],
        OUTPUT_DIR / "multiclass_classification_report.txt",
    )
    save_artifact(
        ARTIFACT_DIR / "multiclass_svm.joblib",
        model=model,
        task_name="multiclass",
        feature_columns=MODEL_FEATURES,
        label_encoder=label_encoder,
        class_names=class_names,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Skin Cancer SVM models")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(CSV_DEFAULT_PATH),
        help="Path to HAM10000_metadata.csv",
    )
    args = parser.parse_args()

    df = load_metadata(args.csv)
    df = add_diagnosis_column(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    binary_metrics = train_binary(df)
    multiclass_metrics = train_multiclass(df)

    summary = {
        "binary_accuracy": binary_metrics["accuracy"],
        "multiclass_accuracy": multiclass_metrics["accuracy"],
        "notes": "Metrics are computed on hold-out test set.",
    }

    (OUTPUT_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Đã huấn luyện xong mô hình.")
    print(f"Binary accuracy     : {binary_metrics['accuracy']:.4f}")
    print(f"Multiclass accuracy : {multiclass_metrics['accuracy']:.4f}")
    print("Artifacts đã lưu trong thư mục artifacts/")
    print("Báo cáo đã lưu trong thư mục outputs/")


if __name__ == "__main__":
    main()
