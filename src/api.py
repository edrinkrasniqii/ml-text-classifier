"""FastAPI endpoint for the email classifier.

Start with: uvicorn src.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.path.join("artifacts", "baseline_model.joblib")

app = FastAPI(title="Email Classifier API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


@app.on_event("startup")
def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first: python src\\train_baseline.py")
    app.state.pipe = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"error": "empty text"}

    pipe = app.state.pipe
    label = pipe.predict([text])[0]
    resp = {"label": label}

    # Include top-3 probabilities when available
    try:
        proba = pipe.predict_proba([text])[0]
        classes = list(pipe.classes_)
        top3 = sorted(
            ({"label": c, "prob": float(p)} for c, p in zip(classes, proba)),
            key=lambda x: x["prob"],
            reverse=True,
        )[:3]
        resp["top3"] = top3
    except Exception:
        # model may not implement predict_proba
        pass

    return resp
