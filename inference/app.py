from fastapi import FastAPI
from schemas import FlowInput
from utils import run_inference


app = FastAPI(title="Intrusion Detection API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def explain(flow: FlowInput):
    data = flow.model_dump()
    return run_inference(data, with_explanation=True)


