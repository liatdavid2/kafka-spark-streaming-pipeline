import time
import os
import mlflow
from mlflow.tracking import MlflowClient
from rollback import rollback_to_previous
import datetime
from drift import mean_drift, ks_drift, population_stability_index
import numpy as np

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
)

MODEL_NAME = "intrusion_model"
CHECK_INTERVAL = 30
ERROR_RATE_THRESHOLD = 0.1


def get_production_metrics(client):
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        return None

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    return run.data.metrics


def check_and_rollback():
    print(f"[MONITOR] {datetime.datetime.now()} running check...")

    client = MlflowClient()

    metrics = get_production_metrics(client)

    if not metrics:
        print("[MONITOR] No production model found")
        return

    error_rate = metrics.get("error_rate", 0.0)

    print(f"[MONITOR] error_rate={error_rate}, threshold={ERROR_RATE_THRESHOLD}")

    if error_rate > ERROR_RATE_THRESHOLD:
        print("[MONITOR] threshold exceeded → rollback")
        rollback_to_previous(client)


if __name__ == "__main__":
    while True:
        check_and_rollback()
        time.sleep(CHECK_INTERVAL)