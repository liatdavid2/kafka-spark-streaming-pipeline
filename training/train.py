import json
from pathlib import Path
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    MODELS_DIR,
    LABEL_COLUMN,
    FEATURE_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
)
from evaluate import evaluate_model
from features import build_features
from utils import make_model_version_path


def load_data() -> pd.DataFrame:

    partition = os.getenv("TRAIN_PARTITION")

    if partition:
        data_path = Path(DATA_PATH) / partition
        print(f"Training on partition: {data_path}")
    else:
        data_path = Path(DATA_PATH)
        print(f"Training on full dataset: {data_path}")

    df = pd.read_parquet(data_path)

    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def save_artifacts(model, metrics: dict, feature_columns: list[str], output_path: Path) -> None:
    joblib.dump(model, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    features_path = output_path.with_suffix(".features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)


def main() -> None:
    partition = os.getenv("TRAIN_PARTITION")
    df = load_data()
    validate_columns(df)

    df = build_features(df)

    feature_columns = FEATURE_COLUMNS + [
        "bytes_total",
        "pkts_total",
        "byte_ratio",
        "pkt_ratio",
        "load_ratio",
        "ttl_diff",
        "jit_total",
        "mean_size_total",
    ]

    X = df[feature_columns]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    metrics["partition"] = partition

    output_path = make_model_version_path(MODELS_DIR)
    save_artifacts(model, metrics, feature_columns, output_path)

    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()