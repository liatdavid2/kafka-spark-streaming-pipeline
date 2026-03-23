from fastapi import FastAPI
from schemas import FlowInput
from features import build_features_from_json
from model import load_model
from rules import evaluate_rules

app = FastAPI(title="Intrusion Detection API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(flow: FlowInput):

    data = flow.model_dump()

    # Step 1 — run rules first
    rule_result = evaluate_rules(data)

    rule_score = rule_result["rule_score"]
    matched_rules = rule_result["matched_rules"]
    rule_actions = rule_result["rule_actions"]
    reasons = rule_result["reasons"]
    attack_types = rule_result["attack_hypothesis"]

    # -------------------------
    # Rule-only decision
    # -------------------------

    if "BLOCK" in rule_actions:
        return {
            "decision": "BLOCK",
            "decision_source": "RULE",
            "rule_score": rule_score,
            "matched_rules": matched_rules,
            "attack_hypothesis": attack_types,
            "reasons": reasons
        }

    if "ALERT" in rule_actions and rule_score >= 0.3:
        return {
            "decision": "ALERT",
            "decision_source": "RULE",
            "rule_score": rule_score,
            "matched_rules": matched_rules,
            "attack_hypothesis": attack_types,
            "reasons": reasons
        }

    # -------------------------
    # Step 2 — call model only if needed
    # -------------------------

    model, threshold = load_model()

    df = build_features_from_json(data)

    try:
        proba = extract_proba(model, df)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    if proba is None:
        return {"error": "Model does not support probability prediction"}

    ml_score = float(proba)

    # -------------------------
    # ML fallback decision
    # -------------------------

    if ml_score >= threshold:
        decision = "BLOCK"
    elif ml_score >= threshold * 0.7:
        decision = "ALERT"
    else:
        decision = "ALLOW"

    return {
        "prediction": int(ml_score > threshold),
        "ml_score": ml_score,
        "rule_score": rule_score,
        "decision": decision,
        "decision_source": "ML",
        "matched_rules": matched_rules,
        "attack_hypothesis": attack_types,
        "reasons": reasons
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