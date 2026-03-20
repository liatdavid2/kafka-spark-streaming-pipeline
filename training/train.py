import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

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


MODEL_NAME = "intrusion_model"


def load_data() -> pd.DataFrame:
    partition = os.getenv("TRAIN_PARTITION")
    base_path = Path(DATA_PATH)

    if partition:
        data_path = base_path / partition
        print(f"Training on partition: {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Partition path does not exist: {data_path}")

        # If partition is an hour, load all hours from the same date
        if "hour=" in partition:
            date_path = data_path.parent
        else:
            date_path = data_path
    else:
        # Fallback: load latest available date
        date_dirs = sorted(base_path.glob("date=*"))
        if not date_dirs:
            raise FileNotFoundError(f"No date partitions found under: {base_path}")
        date_path = date_dirs[-1]

    print(f"Loading full date: {date_path}")

    dfs = []

    for hour_dir in sorted(date_path.glob("hour=*")):
        print(f"Loading {hour_dir}")

        df = pd.read_parquet(hour_dir)

        hour = hour_dir.name.split("=")[1]
        date = date_path.name.split("=")[1]

        df["hour"] = int(hour)
        df["date"] = date

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No hour partitions found under: {date_path}")

    df = pd.concat(dfs, ignore_index=True)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def save_artifacts(
    model: RandomForestClassifier,
    metrics: dict,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    features_path = output_path.with_suffix(".features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)


def should_promote_model(
    metrics: dict,
    latency: float,
    prod_f1: float | None,
) -> tuple[bool, list[str]]:
    reasons = []

    current_f1 = metrics.get("f1", 0.0)
    current_precision = metrics.get("precision", 0.0)
    current_recall = metrics.get("recall", 0.0)

    if current_f1 < 0.90:
        reasons.append(f"F1 too low: {current_f1:.4f} < 0.90")

    if current_precision < 0.85:
        reasons.append(f"Precision too low: {current_precision:.4f} < 0.85")

    if current_recall < 0.85:
        reasons.append(f"Recall too low: {current_recall:.4f} < 0.85")

    if latency > 0.5:
        reasons.append(f"Latency too high: {latency:.4f} > 0.5")

    if prod_f1 is not None and current_f1 <= prod_f1:
        reasons.append(f"Not better than current production: {current_f1:.4f} <= {prod_f1:.4f}")

    return len(reasons) == 0, reasons


def register_and_promote_model(
    client: MlflowClient,
    metrics: dict,
    latency: float,
) -> None:
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

    if not latest_versions:
        print("No new model version found in Registry")
        return

    latest = latest_versions[0]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Model version {latest.version} moved to Staging")

    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    prod_f1 = None

    if prod_versions:
        prod_run_id = prod_versions[0].run_id
        prod_run = client.get_run(prod_run_id)
        prod_f1 = prod_run.data.metrics.get("f1")
        print(f"Current Production F1: {prod_f1}")

    promote, reasons = should_promote_model(
        metrics=metrics,
        latency=latency,
        prod_f1=prod_f1,
    )

    if promote:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model version {latest.version} promoted to Production")
    else:
        print("Model NOT promoted to Production")
        for reason in reasons:
            print(f"Promotion check failed: {reason}")


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

    df = df.sort_values(["hour", "stime"])

    unique_hours = sorted(df["hour"].unique())

    if len(unique_hours) <= 1:
        raise ValueError("Not enough hours for hour-based split")

    train_hours = unique_hours[:-1]
    test_hour = unique_hours[-1]

    train_df = df[df["hour"].isin(train_hours)]
    test_df = df[df["hour"] == test_hour]

    print(f"Train hours: {train_hours}")
    print(f"Test hour: {test_hour}")

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN]

    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN]

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("partition", partition)
        mlflow.log_param("train_hours", str(train_hours))
        mlflow.log_param("test_hour", str(test_hour))

        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("feature_version", "v1")
        mlflow.set_tag("data_partition", partition or "full")
        mlflow.set_tag("split_type", "hour_based")
        mlflow.set_tag("monitoring_mode", "offline")

        model = train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics["partition"] = partition

        start = time.time()
        _ = model.predict(X_test)
        latency = time.time() - start
        mlflow.log_metric("inference_latency_sec", latency)

        error_rate = float((y_pred != y_test).mean())
        mlflow.log_metric("error_rate", error_rate)

        unique_preds, counts = np.unique(y_pred, return_counts=True)
        for pred_class, count in zip(unique_preds, counts):
            mlflow.log_metric(f"pred_class_{pred_class}", int(count))

        for col in feature_columns:
            drift = abs(X_train[col].mean() - X_test[col].mean())
            mlflow.log_metric(f"drift_{col}", float(drift))

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        model_path = "model.joblib"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        client = MlflowClient()
        register_and_promote_model(
            client=client,
            metrics=metrics,
            latency=latency,
        )

        metrics_path = "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        features_path = "features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_columns, f, indent=2)
        mlflow.log_artifact(features_path)

    output_path = make_model_version_path(MODELS_DIR)
    save_artifacts(model, metrics, feature_columns, output_path)

    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()