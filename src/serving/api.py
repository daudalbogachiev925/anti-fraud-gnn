"""FastAPI serving endpoint."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from pathlib import Path
from src.models.tabular import TabularFraudModel

app = FastAPI(title="Fraud Detection API", version="0.1.0")

MODEL_PATH = Path("models/catboost_fraud.cbm")
model: TabularFraudModel | None = None


class FraudRequest(BaseModel):
    features: List[float]


class FraudResponse(BaseModel):
    fraud_probability: float
    decision: str


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = TabularFraudModel.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: no model at {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=FraudResponse)
def predict(req: FraudRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    x = np.array([req.features])
    prob = float(model.predict_proba(x)[0])
    decision = "block" if prob > 0.8 else "review" if prob > 0.5 else "allow"
    return FraudResponse(fraud_probability=prob, decision=decision)
