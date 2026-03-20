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

    # change: load threshold too
    model, threshold = load_model()

    data = flow.model_dump()
    df = build_features_from_json(data)

    # use robust proba extraction
    try:
        proba = extract_proba(model, df)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # fallback (prevents null)
    if proba is None:
        return {"error": "Model does not support probability prediction"}

    # change: prediction based on threshold (not model.predict)
    prediction = int(proba > threshold)

    decision = "ALLOW"

    # keep your logic but aligned with threshold
    if proba > threshold:
        decision = "BLOCK"
    elif proba > threshold * 0.7:
        decision = "ALERT"
    elif prediction == 1:
        decision = "ALERT"

    print(f"[PREDICT] pred={prediction}, proba={proba}, threshold={threshold}, decision={decision}")

    return {
        "prediction": int(prediction),
        "risk_score": float(proba),
        "decision": decision
    }


def extract_proba(model, df):
    # direct model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        return probs[1] if len(probs) > 1 else probs[0]

    # sklearn Pipeline
    if hasattr(model, "steps"):
        last_model = model.steps[-1][1]
        if hasattr(last_model, "predict_proba"):
            probs = last_model.predict_proba(df)[0]
            return probs[1] if len(probs) > 1 else probs[0]

    # fallback
    return None