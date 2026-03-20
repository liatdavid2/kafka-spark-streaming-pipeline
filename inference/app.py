from fastapi import FastAPI
from schemas import FlowInput
from model import model
from features import build_features_from_json

app = FastAPI(title="Intrusion Detection API")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(flow: FlowInput):

    data = flow.dict()

    df = build_features_from_json(data)

    prediction = model.predict(df)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0][1]

    decision = "ALLOW"
    if proba:
        if proba > 0.9:
            decision = "BLOCK"
        elif proba > 0.7:
            decision = "ALERT"

    return {
        "prediction": int(prediction),
        "risk_score": float(proba) if proba else None,
        "decision": decision
    }