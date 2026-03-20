import os
import mlflow.pyfunc

MODEL_URI = "models:/intrusion_model/Production"


def load_model():
    print(f"Loading model from MLflow: {MODEL_URI}")
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_model()