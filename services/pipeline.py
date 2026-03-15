"""Service layer wrapping existing pipelines for API/UI use."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

import run_all as ra
from hybrid_ddi_pipeline import HybridConfig, run_hybrid_cv

ROOT = Path(__file__).resolve().parent.parent
RUN_LOG = ROOT / "runs" / "runs.jsonl"
ARTIFACT_DIR = ROOT / "model_artifacts"

# Simple module-level caches to avoid reloading large files repeatedly.
_CACHED_MODELS = None
_CACHED_IMPUTER = None
_CACHED_FEATURE_COLS: List[str] = []
_CACHED_PAIR_FEATURES = None
_CACHED_LOOKUP = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_run_dir():
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)


def log_run(record: Dict[str, Any]) -> None:
    _ensure_run_dir()
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    if not RUN_LOG.exists():
        return []
    lines = RUN_LOG.read_text().strip().splitlines()
    if not lines:
        return []
    records = [json.loads(l) for l in lines]
    return records[-limit:][::-1]


def list_artifacts(output_dir: Path) -> List[Dict[str, Any]]:
    if not output_dir.exists():
        return []
    out = []
    for p in sorted(output_dir.glob("*")):
        if p.is_file():
            stat = p.stat()
            out.append({
                "name": p.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return out


def load_pair_features(output_dir: Path = ARTIFACT_DIR) -> pd.DataFrame:
    global _CACHED_PAIR_FEATURES
    if _CACHED_PAIR_FEATURES is not None:
        return _CACHED_PAIR_FEATURES
    pf_path = output_dir / "pair_features.parquet"
    if not pf_path.exists():
        return pd.DataFrame()
    _CACHED_PAIR_FEATURES = pd.read_parquet(pf_path)
    return _CACHED_PAIR_FEATURES


def load_drug_lookup(output_dir: Path = ARTIFACT_DIR) -> Dict[str, str]:
    global _CACHED_LOOKUP
    if _CACHED_LOOKUP is not None:
        return _CACHED_LOOKUP
    lk_path = output_dir / "drug_name_lookup.json"
    if not lk_path.exists():
        return {}
    _CACHED_LOOKUP = json.loads(lk_path.read_text())
    return _CACHED_LOOKUP


def ensure_feature_columns(pf: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    pf = pf.copy()
    for col in feature_cols:
        if col not in pf.columns:
            pf[col] = 0.0
    return pf[feature_cols]


def compute_shap(model, X: Optional[pd.DataFrame], feature_cols: List[str], sample_size: int = 300) -> Dict[str, Any]:
    if X is None or len(X) == 0:
        return {"global_top": [], "local_top": []}
    n = min(sample_size, len(X))
    X_sample = X.sample(n, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    shap_arr = np.asarray(shap_vals)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[1]
    mean_abs = np.abs(shap_arr).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_top = [
        {"feature": feature_cols[i], "shap": float(mean_abs[i])}
        for i in order[:10]
    ]

    proba = model.predict_proba(X_sample)[:, 1]
    idx = int(np.argmax(proba))
    local_vals = shap_arr[idx]
    local_order = np.argsort(np.abs(local_vals))[::-1][:10]
    local_top = [
        {
            "feature": feature_cols[i],
            "shap": float(local_vals[i]),
            "value": float(X_sample.iloc[idx, i]),
        }
        for i in local_order
    ]
    return {"global_top": global_top, "local_top": local_top}


def load_artifacts(output_dir: Path):
    global _CACHED_MODELS, _CACHED_IMPUTER, _CACHED_FEATURE_COLS
    if _CACHED_MODELS is not None and _CACHED_IMPUTER is not None and _CACHED_FEATURE_COLS:
        return _CACHED_MODELS, _CACHED_IMPUTER, _CACHED_FEATURE_COLS

    models = {
        "logreg": joblib.load(output_dir / "logreg.pkl"),
        "random_forest": joblib.load(output_dir / "random_forest.pkl"),
        "xgboost": joblib.load(output_dir / "xgboost.pkl"),
    }
    imputer = joblib.load(output_dir / "imputer.pkl")
    feature_cols = json.loads((output_dir / "feature_cols.json").read_text())
    _CACHED_MODELS, _CACHED_IMPUTER, _CACHED_FEATURE_COLS = models, imputer, feature_cols
    return models, imputer, feature_cols


def sample_predictions(output_dir: Path, top_k: int = 20) -> List[Dict[str, Any]]:
    needed = [output_dir / "xgboost.pkl", output_dir / "feature_cols.json", output_dir / "pair_features.parquet", output_dir / "imputer.pkl"]
    if not all(p.exists() for p in needed):
        return []

    models, imputer, feature_cols = load_artifacts(output_dir)
    pf = load_pair_features(output_dir)
    pf_prepped = ensure_feature_columns(pf, feature_cols)
    X_imp = pd.DataFrame(imputer.transform(pf_prepped), columns=feature_cols)
    proba = models["xgboost"].predict_proba(X_imp)[:, 1]
    pf = pf.copy()
    pf["proba_xgb"] = proba
    pf = pf.sort_values("proba_xgb", ascending=False).head(top_k)
    return pf.assign(proba_xgb=pf["proba_xgb"].astype(float)).to_dict(orient="records")


def run_pipeline(
    dev_mode: bool = False,
    folds: int = 3,
    shap_samples: int = 500,
    ndd_mode: str = "mock",
    ndd_model_path: Optional[str] = None,
    run_hybrid: bool = True,
) -> Dict[str, Any]:
    root = ROOT
    args = {
        "offsides": root / "data" / "OFFSIDES.csv.gz",
        "twosides": root / "data" / "TWOSIDES.csv.gz",
        "output": root / "model_artifacts",
        "data_dir": root / "NDD" / "DS1",
        "folds": folds,
        "shap_samples": shap_samples,
        "ndd_mode": ndd_mode,
        "ndd_model_path": ndd_model_path or (root / "ddi_model.h5"),
    }

    folds_eff = min(folds, 2) if dev_mode else folds
    shap_eff = min(shap_samples, 150) if dev_mode else shap_samples
    chunk_eff = 120_000 if dev_mode else 300_000

    start = time.time()

    models, dataset, imputer, classical_metrics, X_test, y_test = ra.train_classical(
        args["offsides"], args["twosides"], args["output"], chunksize=chunk_eff, dev_mode=dev_mode
    )

    if X_test is not None and imputer is not None and dataset is not None:
        X_test = pd.DataFrame(imputer.transform(X_test), columns=dataset.feature_cols)

    shap_info = {}
    xgb_model = models.get("xgboost")
    if xgb_model is not None and dataset is not None:
        shap_info = compute_shap(xgb_model, X_test, dataset.feature_cols, sample_size=shap_eff)

    hybrid_metrics: Dict[str, Any] = {"skipped": True}
    if run_hybrid:
        hybrid_cfg = HybridConfig(
            data_dir=str(args["data_dir"]),
            n_splits=folds_eff,
            random_state=42,
            ndd_mode=ndd_mode,
            ndd_model_path=str(args["ndd_model_path"]) if args["ndd_model_path"] else None,
        )
        hybrid_metrics = run_hybrid_cv(hybrid_cfg)

    ndd_status = "skipped"
    try:
        ra.run_ndd_smoke()
        ndd_status = "ok"
    except Exception as exc:  # pragma: no cover
        ndd_status = f"failed: {exc}"

    duration = time.time() - start

    result = {
        "run_id": _now_iso(),
        "mode": "dev" if dev_mode else "full",
        "params": {
            "folds": folds_eff,
            "shap_samples": shap_eff,
            "chunk_size": chunk_eff,
            "ndd_mode": ndd_mode,
        },
        "timing_sec": duration,
        "classical_metrics": classical_metrics,
        "hybrid_metrics": hybrid_metrics,
        "shap": shap_info,
        "artifacts": list_artifacts(args["output"]),
        "ndd_status": ndd_status,
    }

    log_run(result)
    return result


def run_dev() -> Dict[str, Any]:
    # Keep dev runs responsive: smaller shap sample, skip hybrid fusion
    return run_pipeline(dev_mode=True, folds=1, shap_samples=75, ndd_mode="mock", run_hybrid=False)


def run_full() -> Dict[str, Any]:
    return run_pipeline(dev_mode=False, folds=3, shap_samples=500, ndd_mode="mock")


# ---------- Prediction & explainability helpers ----------


def _risk_label(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Moderate"
    return "High"


def find_pair(drug_a_id: str, drug_b_id: str, pf: pd.DataFrame) -> Optional[pd.Series]:
    # Try direct then swapped
    direct = pf[(pf["drug_1_rxnorm_id"].astype(str) == str(drug_a_id)) & (pf["drug_2_rxnorm_id"].astype(str) == str(drug_b_id))]
    if len(direct):
        return direct.iloc[0]
    swapped = pf[(pf["drug_1_rxnorm_id"].astype(str) == str(drug_b_id)) & (pf["drug_2_rxnorm_id"].astype(str) == str(drug_a_id))]
    if len(swapped):
        return swapped.iloc[0]
    return None


def predict_interaction(drug_a_id: str, drug_b_id: str) -> Dict[str, Any]:
    models, imputer, feature_cols = load_artifacts(ARTIFACT_DIR)
    pf = load_pair_features(ARTIFACT_DIR)
    lookup = load_drug_lookup(ARTIFACT_DIR)

    if pf.empty:
        raise RuntimeError("pair_features.parquet not found")

    row = find_pair(drug_a_id, drug_b_id, pf)
    if row is None:
        raise ValueError("Pair not found in cached features")

    row_df = row.to_frame().T
    feats = ensure_feature_columns(row_df, feature_cols)
    feats_imp = pd.DataFrame(imputer.transform(feats), columns=feature_cols)

    proba = {
        name: float(model.predict_proba(feats_imp)[:, 1][0])
        for name, model in models.items()
    }

    # Simple hybrid = average of available classical probs
    hybrid_prob = float(np.mean(list(proba.values())))

    agreement = sum(p >= 0.5 for p in proba.values())
    total_models = len(proba)

    shap_info = compute_shap(models.get("xgboost"), feats_imp, feature_cols, sample_size=1)

    return {
        "drug_a_id": str(drug_a_id),
        "drug_b_id": str(drug_b_id),
        "drug_a_name": lookup.get(str(drug_a_id), str(drug_a_id)),
        "drug_b_name": lookup.get(str(drug_b_id), str(drug_b_id)),
        "proba": proba,
        "hybrid_prob": hybrid_prob,
        "risk": _risk_label(hybrid_prob),
        "confidence": float(agreement / total_models) if total_models else 0.0,
        "agreement": f"{agreement}/{total_models}",
        "shap": shap_info,
        "features": {k: float(v) for k, v in row.items() if k in feats.columns},
    }


def list_drugs() -> List[Dict[str, str]]:
    lookup = load_drug_lookup(ARTIFACT_DIR)
    return [{"id": k, "name": v} for k, v in lookup.items()]


def dataset_pairs(sample: int = 100) -> List[Dict[str, Any]]:
    pf = load_pair_features(ARTIFACT_DIR)
    lookup = load_drug_lookup(ARTIFACT_DIR)
    if pf.empty:
        return []
    pf = pf.head(sample).copy()
    pf["drug1_name"] = pf["drug_1_rxnorm_id"].astype(str).map(lookup).fillna(pf["drug_1_rxnorm_id"].astype(str))
    pf["drug2_name"] = pf["drug_2_rxnorm_id"].astype(str).map(lookup).fillna(pf["drug_2_rxnorm_id"].astype(str))
    return pf.to_dict(orient="records")


__all__ = [
    "run_dev",
    "run_full",
    "run_pipeline",
    "list_runs",
    "list_artifacts",
    "sample_predictions",
]
