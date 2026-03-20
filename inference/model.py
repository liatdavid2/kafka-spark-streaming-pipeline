from pathlib import Path
import joblib

MODELS_DIR = Path("/app/models")


def load_model():
    # find all model files recursively
    models = list(MODELS_DIR.rglob("*.joblib"))

    if not models:
        raise FileNotFoundError("No model found in /app/models")

    # pick latest by folder name (timestamp)
    latest_model = sorted(models)[-1]

    print(f"Loading model: {latest_model}")

    return joblib.load(latest_model)


model = load_model()