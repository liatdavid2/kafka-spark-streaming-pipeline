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

    # use robust proba extraction
    proba = extract_proba(model, df)

    # fallback (prevents null)
    if proba is None:
        proba = float(prediction)

    decision = "ALLOW"

    if proba > 0.9:
        decision = "BLOCK"
    elif proba > 0.6:
        decision = "ALERT"
    elif prediction == 1:
        decision = "ALERT"

    print(f"[PREDICT] pred={prediction}, proba={proba}, decision={decision}")

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