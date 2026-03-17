from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from services.pipeline import (
    dataset_pairs,
    list_artifacts,
    list_drugs,
    list_runs,
    predict_interaction,
    run_dev,
    run_full,
    run_pipeline,
    sample_predictions,
)

app = FastAPI(title="DDI Hybrid Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs/dev")
def run_dev_endpoint():
    return run_dev()


@app.post("/runs/full")
def run_full_endpoint():
    return run_full()


@app.post("/runs")
def run_custom(
    dev_mode: bool = False,
    folds: int = 3,
    shap_samples: int = 500,
    ndd_mode: str = "mock",
    ndd_model_path: Optional[str] = None,
):
    return run_pipeline(dev_mode=dev_mode, folds=folds, shap_samples=shap_samples, ndd_mode=ndd_mode, ndd_model_path=ndd_model_path)


@app.get("/runs")
def get_runs(limit: int = Query(20, ge=1, le=200)):
    return list_runs(limit)


@app.get("/runs/latest")
def get_latest():
    runs = list_runs(1)
    return runs[0] if runs else {}


@app.get("/artifacts")
def get_artifacts(output: str = "model_artifacts"):
    return list_artifacts(Path(output))


@app.get("/predictions/sample")
def get_sample_predictions(top_k: int = Query(20, ge=1, le=200), output: str = "model_artifacts"):
    return sample_predictions(Path(output), top_k=top_k)


@app.get("/drugs")
def get_drugs():
    return list_drugs()


@app.get("/dataset/pairs")
def get_pairs(sample: Optional[int] = Query(None, ge=1)):
    return dataset_pairs(sample=sample)


@app.post("/predict")
def post_predict(drug_a_id: str, drug_b_id: str):
    try:
        return predict_interaction(drug_a_id, drug_b_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
