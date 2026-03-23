from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_utils import load_artifact


def predict_from_metadata(
    artifact_path: str | Path,
    age: float | int | None,
    sex: str,
    localization: str,
) -> dict:
    payload = load_artifact(artifact_path)
    model = payload["model"]
    label_encoder = payload.get("label_encoder")
    task_name = payload["task_name"]

    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "localization": localization,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probabilities = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        probabilities = [float(value) for value in proba]

    if task_name == "multiclass" and label_encoder is not None:
        label_name = label_encoder.inverse_transform([prediction])[0]
    else:
        label_name = int(prediction)

    return {
        "task_name": task_name,
        "prediction": label_name,
        "probabilities": probabilities,
        "class_names": payload.get("class_names"),
    }
