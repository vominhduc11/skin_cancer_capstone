from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import (
    BINARY_BENIGN,
    BINARY_HIGH_RISK,
    CATEGORICAL_FEATURES,
    DX_FULLNAME,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SUPPORTED_MULTICLASS_LABELS,
)


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load HAM10000 metadata CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"dx", "age", "sex", "localization", "image_id", "lesion_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "File CSV thiếu các cột bắt buộc: " + ", ".join(sorted(missing_columns))
        )

    return df.copy()


def add_diagnosis_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map short dx code to readable diagnosis name."""
    df = df.copy()
    df["diagnosis"] = df["dx"].map(DX_FULLNAME).fillna(df["dx"])
    return df


def build_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary dataset based on the assignment mapping:
    high-risk = mel, bcc, akiec, vasc -> 1
    benign = nv, df, bkl -> 0
    """
    df = df.copy()
    valid_labels = BINARY_HIGH_RISK | BINARY_BENIGN
    df = df[df["dx"].isin(valid_labels)].copy()

    df["target_binary"] = df["dx"].apply(lambda value: 1 if value in BINARY_HIGH_RISK else 0)
    return df


def build_multiclass_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the 7 supported classes."""
    df = df.copy()
    df = df[df["dx"].isin(SUPPORTED_MULTICLASS_LABELS)].copy()
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only features used for training."""
    return df[MODEL_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_svm_model(probability: bool = False) -> Pipeline:
    """Create a full sklearn pipeline with preprocessing + SVM."""
    preprocessor = build_preprocessor()

    classifier = SVC(
        kernel="rbf",
        C=3.0,
        gamma="scale",
        class_weight="balanced",
        probability=probability,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return model


def evaluate_predictions(
    y_true,
    y_pred,
    class_names: list[str],
) -> dict[str, Any]:
    """Return evaluation metrics in dict form."""
    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "classification_report_text": report_text,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def save_confusion_matrix(
    cm,
    class_names: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Save confusion matrix as image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_text_report(report_text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def save_artifact(
    artifact_path: str | Path,
    model: Pipeline,
    task_name: str,
    feature_columns: list[str],
    label_encoder: LabelEncoder | None = None,
    class_names: list[str] | None = None,
) -> None:
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_name": task_name,
        "feature_columns": feature_columns,
        "model": model,
        "class_names": class_names,
        "label_encoder": label_encoder,
    }
    joblib.dump(payload, artifact_path)


def load_artifact(artifact_path: str | Path) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file model: {artifact_path}")
    return joblib.load(artifact_path)
