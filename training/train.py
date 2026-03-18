import json
from pathlib import Path
import os
import time
import numpy as np

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

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
    base_path = Path(DATA_PATH)

    if partition:
        data_path = base_path / partition
        print(f"Training on partition: {data_path}")

        # If partition is an hour, load all hours from the same date
        if "hour=" in partition:
            date_path = data_path.parent
        else:
            date_path = data_path

    else:
        # Fallback: load latest available date
        date_dirs = sorted(base_path.glob("date=*"))
        date_path = date_dirs[-1]

    print(f"Loading full date: {date_path}")

    dfs = []

    # Iterate over all hour partitions
    for hour_dir in sorted(date_path.glob("hour=*")):
        print(f"Loading {hour_dir}")

        df = pd.read_parquet(hour_dir)

        # Extract hour and date from directory structure
        hour = hour_dir.name.split("=")[1]
        date = date_path.name.split("=")[1]

        df["hour"] = int(hour)
        df["date"] = date

        dfs.append(df)

    # Combine all partitions into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)
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
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("intrusion-detection")
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

    #  sort by time 
    df = df.sort_values("stime") 

    # split by hour

    df = df.sort_values(["hour", "stime"])

    unique_hours = sorted(df["hour"].unique())

    if len(unique_hours) > 1:
        train_hours = unique_hours[:-1]
        test_hour = unique_hours[-1]

        train_df = df[df["hour"].isin(train_hours)]
        test_df = df[df["hour"] == test_hour]

        print(f"Train hours: {train_hours}")
        print(f"Test hour: {test_hour}")
    else:
        raise ValueError("Not enough hours for split")

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN]

    # MLflow tracking
    with mlflow.start_run():

        # params
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("partition", partition)

        # tags (metadata)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("feature_version", "v1")
        mlflow.set_tag("data_partition", partition or "full")

        mlflow.set_tag("split_type", "hour_based")
        mlflow.set_tag("monitoring_mode", "offline")
        mlflow.log_param("train_hours", str(train_hours))
        mlflow.log_param("test_hour", str(test_hour))

        # dataset info
        #mlflow.log_metric("dataset_size", len(df))
        #mlflow.log_metric("train_size", len(X_train))
        #mlflow.log_metric("test_size", len(X_test))

        # train model
        model = train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        # -------------------------
        # Latency
        # -------------------------
        start = time.time()
        _ = model.predict(X_test)
        latency = time.time() - start
        mlflow.log_metric("inference_latency_sec", latency)

        # -------------------------
        # Error rate
        # -------------------------
        error_rate = (y_pred != y_test).mean()
        mlflow.log_metric("error_rate", error_rate)

        # -------------------------
        # Prediction distribution
        # -------------------------
        unique, counts = np.unique(y_pred, return_counts=True)
        for k, v in zip(unique, counts):
            mlflow.log_metric(f"pred_class_{k}", int(v))

        # -------------------------
        # Data drift (simple)
        # -------------------------
        for col in feature_columns:
            drift = abs(X_train[col].mean() - X_test[col].mean())
            mlflow.log_metric(f"drift_{col}", drift)
            
        metrics["partition"] = partition

        # metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # save model locally
        model_path = "model.joblib"
        joblib.dump(model, model_path)

        # log model as artifact
        mlflow.log_artifact(model_path)

        # save + log metrics file
        metrics_path = "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # save + log features
        features_path = "features.json"
        with open(features_path, "w") as f:
            json.dump(feature_columns, f, indent=2)
        mlflow.log_artifact(features_path)

    # keep local versioning (important)
    output_path = make_model_version_path(MODELS_DIR)
    save_artifacts(model, metrics, feature_columns, output_path)

    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()