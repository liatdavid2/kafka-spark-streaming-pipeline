import time
import os
import mlflow
from mlflow.tracking import MlflowClient
from rollback import rollback_to_previous
import datetime
from drift import mean_drift, ks_drift, population_stability_index
import numpy as np
import json
import pandas as pd
from pathlib import Path

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
)

MODEL_NAME = "intrusion_model"
CHECK_INTERVAL = 30
ERROR_RATE_THRESHOLD = 0.1

DATA_PATH = Path("/app/output")  # parquet from Spark


# -----------------------------
# Load train distribution from MLflow
# -----------------------------
def get_train_stats(client):
    try:
        # -----------------------------
        # Get latest experiment run (NOT model)
        # -----------------------------
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

        if not versions:
            print("[DRIFT] No production model")
            return None

        run_id = versions[0].run_id
        print(f"[DEBUG] Using production run_id = {run_id}")

        # -----------------------------
        # Find artifact
        # -----------------------------
        artifact_path = "drift/train_stats.json"

        try:
            artifacts = client.download_artifacts(run_id, artifact_path)
            print(f"[DRIFT] Loaded from {artifact_path}")
        except Exception as e:
            print(f"[DRIFT] train_stats.json not found: {e}")
            return None
        print("[DEBUG drift files]", [a.path for a in artifacts])

        target = None
        for a in artifacts:
            if a.path.endswith("train_stats.json"):
                target = a.path
                break

        if not target:
            print("[DRIFT] train_stats.json not found")
            return None

        # -----------------------------
        # Download
        # -----------------------------
        local_path = client.download_artifacts(run_id, target)

        with open(local_path, "r") as f:
            stats = json.load(f)

        print(f"[DRIFT] Loaded train stats from run {run_id}")
        return stats

    except Exception as e:
        print(f"[DRIFT ERROR] {e}")
        return None


# -----------------------------
# Load current production data
# -----------------------------
def load_current_data():
    files = sorted(DATA_PATH.rglob("*.parquet"))

    if not files:
        print("[DATA] No parquet files found")
        return None

    latest = files[-1]
    print(f"[DATA] loading current data from {latest}")

    df = pd.read_parquet(latest)
    return df


# -----------------------------
# MLflow metrics
# -----------------------------
def get_production_metrics(client):
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        return None

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    return run.data.metrics


# -----------------------------
# Drift computation
# -----------------------------
def compute_drift(client):
    train_stats = get_train_stats(client)

    if train_stats is None:
        print("[DRIFT] skipping")
        return False

    current_df = load_current_data()

    if current_df is None:
        print("[DRIFT] No current data → skipping")
        return False

    drift_results = []

    print("\n[DRIFT CHECK PER FEATURE]")

    for col, train_info in train_stats.items():
        if col not in current_df.columns:
            continue

        current_values = current_df[col].dropna().values
        train_values = np.array(train_info["sample"])

        if len(current_values) < 50:
            continue

        # Mean
        mean_flag, mean_diff = mean_drift(
            train_info["mean"],
            np.mean(current_values)
        )

        # KS
        ks_flag, p_value = ks_drift(train_values, current_values)

        # PSI
        psi_flag, psi_value = population_stability_index(
            train_values,
            current_values
        )

        print(f"\nFeature: {col}")
        print(f"  Mean diff: {mean_diff:.4f} → drift={mean_flag}")
        print(f"  KS p-value: {p_value:.6f} → drift={ks_flag}")
        print(f"  PSI: {psi_value:.4f} → drift={psi_flag}")

        votes = sum([mean_flag, ks_flag, psi_flag])
        feature_drift = votes >= 2

        print(f"  → FINAL: drift={feature_drift} ({votes}/3)")

        drift_results.append(feature_drift)

    if not drift_results:
        print("[DRIFT] No features evaluated")
        return False

    total_drift = sum(drift_results) > len(drift_results) * 0.3

    print(f"\n[DRIFT SUMMARY] {sum(drift_results)}/{len(drift_results)} features drifted")
    print(f"[DRIFT FINAL] DRIFT={total_drift}")

    return total_drift


# -----------------------------
# Main check
# -----------------------------
def check_and_rollback():
    print(f"\n[MONITOR] {datetime.datetime.now()} running check...")

    client = MlflowClient()

    metrics = get_production_metrics(client)

    if not metrics:
        print("[MONITOR] No production model found")
        return

    error_rate = metrics.get("error_rate", 0.0)

    print(f"[MONITOR] error_rate={error_rate}, threshold={ERROR_RATE_THRESHOLD}")

    # -------- DRIFT --------
    drift_detected = compute_drift(client)

    # -------- DECISION --------
    if error_rate > ERROR_RATE_THRESHOLD or drift_detected:
        print("\n[DECISION] ISSUE DETECTED")

        if error_rate > ERROR_RATE_THRESHOLD:
            print("Reason: High error rate")

        if drift_detected:
            print("Reason: Data drift detected")

        print("→ ACTION: ROLLBACK\n")
        rollback_to_previous(client)

    else:
        print("\n[DECISION] System stable")
        print("→ ACTION: NO ROLLBACK\n")


# -----------------------------
# Loop
# -----------------------------
if __name__ == "__main__":
    while True:
        check_and_rollback()
        time.sleep(CHECK_INTERVAL)