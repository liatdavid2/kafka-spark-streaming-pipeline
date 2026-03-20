from fastapi import FastAPI
from schemas import FlowInput
from features import build_features_from_json
from model import load_model

app = FastAPI(title="Intrusion Detection API")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(flow: FlowInput):

    model = load_model() 

    data = flow.model_dump()
    df = build_features_from_json(data)

    try:
        prediction = model.predict(df)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        if len(probs) > 1:
            proba = probs[1]

    decision = "ALLOW"

    if proba is not None:
        if proba > 0.9:
            decision = "BLOCK"
        elif proba > 0.6:
            decision = "ALERT"
        elif prediction == 1:
            decision = "ALERT"
    else:
        if prediction == 1:
            decision = "ALERT"

    print(f"[PREDICT] pred={prediction}, proba={proba}, decision={decision}")

    return {
        "prediction": int(prediction),
        "risk_score": float(proba) if proba is not None else None,
        "decision": decision
    }