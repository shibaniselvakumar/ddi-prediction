"""Service layer wrapping existing pipelines for API/UI use."""
from __future__ import annotations

import json
import hashlib
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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    return f if np.isfinite(f) else default


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


def _drug_profile(drug_id: str, pf: pd.DataFrame) -> Dict[str, float]:
    d = str(drug_id)
    mask = (pf["drug_1_rxnorm_id"].astype(str) == d) | (pf["drug_2_rxnorm_id"].astype(str) == d)
    rows = pf[mask]

    def s(col: str) -> pd.Series:
        return pd.to_numeric(rows.get(col, pd.Series(dtype=float)), errors="coerce")

    if rows.empty:
        # Global prior + deterministic id-specific scaling so unseen drugs don't collapse to identical vectors.
        g_avg_prr = _safe_float(pd.to_numeric(pf.get("pair_avg_prr", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 0.0)
        g_max_prr = _safe_float(pd.to_numeric(pf.get("pair_max_prr", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 0.0)
        g_prr_std = _safe_float(pd.to_numeric(pf.get("pair_prr_std", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 1.0)
        g_num_effects = _safe_float(pd.to_numeric(pf.get("pair_num_effects", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 1.0)
        g_rep = _safe_float(pd.to_numeric(pf.get("pair_avg_rep_freq", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 0.0)
        g_a = _safe_float(pd.to_numeric(pf.get("pair_A_sum", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 0.0)
        g_b = _safe_float(pd.to_numeric(pf.get("pair_B_sum", pd.Series(dtype=float)), errors="coerce").mean(skipna=True), 1.0)

        h = int(hashlib.sha256(d.encode("utf-8")).hexdigest()[:8], 16)
        scale = 1.0 + (((h % 2001) - 1000) / 10000.0)  # ~[0.9, 1.1]

        avg_prr = max(g_avg_prr * scale, 0.0)
        max_prr = max(g_max_prr * scale, 0.0)
        prr_std = max(g_prr_std, 1e-6)
        num_side_effects = max(g_num_effects * scale, 1.0)
        mean_rep_freq = max(g_rep * scale, 0.0)
        total_A = max(g_a * scale, 0.0)
        total_B = max(g_b * scale, 1.0)

        return {
            "avg_prr": _safe_float(avg_prr, 0.0),
            "max_prr": _safe_float(max_prr, 0.0),
            "prr_std": _safe_float(prr_std, 1.0),
            "num_side_effects": _safe_float(num_side_effects, 1.0),
            "mean_rep_freq": _safe_float(mean_rep_freq, 0.0),
            "total_A": _safe_float(total_A, 0.0),
            "total_B": _safe_float(total_B, 1.0),
            "snr": _safe_float(avg_prr / max(prr_std, 1e-6), 0.0),
            "n_pairs": 0.0,
        }

    avg_prr = _safe_float(s("pair_avg_prr").mean(skipna=True), 0.0)
    max_prr = _safe_float(s("pair_max_prr").max(skipna=True), 0.0)
    prr_std = _safe_float(s("pair_prr_std").mean(skipna=True), 1.0)
    num_side_effects = _safe_float(s("pair_num_effects").mean(skipna=True), 1.0)
    mean_rep_freq = _safe_float(s("pair_avg_rep_freq").mean(skipna=True), 0.0)
    total_A = _safe_float(s("pair_A_sum").mean(skipna=True), 0.0)
    total_B = _safe_float(s("pair_B_sum").mean(skipna=True), 1.0)

    return {
        "avg_prr": avg_prr,
        "max_prr": max_prr,
        "prr_std": max(prr_std, 1e-6),
        "num_side_effects": max(num_side_effects, 1.0),
        "mean_rep_freq": mean_rep_freq,
        "total_A": total_A,
        "total_B": max(total_B, 1.0),
        "snr": _safe_float(avg_prr / max(prr_std, 1e-6), 0.0),
        "n_pairs": float(len(rows)),
    }


def _synthesize_pair_row(drug_a_id: str, drug_b_id: str, pf: pd.DataFrame) -> pd.Series:
    p1 = _drug_profile(drug_a_id, pf)
    p2 = _drug_profile(drug_b_id, pf)

    w1 = max(p1.get("n_pairs", 0.0), 1.0)
    w2 = max(p2.get("n_pairs", 0.0), 1.0)
    wsum = w1 + w2

    pair_avg_prr = float((p1["avg_prr"] * w1 + p2["avg_prr"] * w2) / wsum)
    pair_max_prr = float(max(p1["max_prr"], p2["max_prr"]))
    pair_prr_std = float((p1["prr_std"] * w1 + p2["prr_std"] * w2) / wsum)
    pair_num_effects = float(max(1.0, (p1["num_side_effects"] + p2["num_side_effects"]) / 2.0))
    pair_A_sum = float((p1["total_A"] * w1 + p2["total_A"] * w2) / wsum)
    pair_B_sum = float((p1.get("total_B", 1.0) * w1 + p2.get("total_B", 1.0) * w2) / wsum)
    pair_avg_prr_error = float(pair_prr_std / max(pair_num_effects ** 0.5, 1.0))
    pair_avg_rep_freq = float((p1["mean_rep_freq"] + p2["mean_rep_freq"]) / 2.0)
    log_pair_avg_prr = float(np.log1p(max(pair_avg_prr, 0.0)))
    log_pair_max_prr = float(np.log1p(max(pair_max_prr, 0.0)))
    pair_snr = float(pair_avg_prr / max(pair_prr_std, 1e-6))
    relative_risk_ratio = float((p1["avg_prr"] + 1e-6) / (p2["avg_prr"] + 1e-6))
    report_confidence = float(pair_num_effects / max(pair_num_effects + pair_B_sum, 1.0))
    effect_diversity_ratio = float(pair_num_effects / max(p1["num_side_effects"] + p2["num_side_effects"], 1.0))

    return pd.Series({
        "drug_1_rxnorm_id": str(drug_a_id),
        "drug_2_rxnorm_id": str(drug_b_id),
        "pair_avg_prr": pair_avg_prr,
        "pair_max_prr": pair_max_prr,
        "pair_prr_std": pair_prr_std,
        "pair_num_effects": pair_num_effects,
        "pair_A_sum": pair_A_sum,
        "pair_B_sum": pair_B_sum,
        "pair_avg_prr_error": pair_avg_prr_error,
        "pair_avg_rep_freq": pair_avg_rep_freq,
        "drug1_avg_prr": p1["avg_prr"],
        "drug1_max_prr": p1["max_prr"],
        "drug1_prr_std": p1["prr_std"],
        "drug1_num_side_effects": p1["num_side_effects"],
        "drug1_mean_rep_freq": p1["mean_rep_freq"],
        "drug1_total_A": p1["total_A"],
        "drug2_avg_prr": p2["avg_prr"],
        "drug2_max_prr": p2["max_prr"],
        "drug2_prr_std": p2["prr_std"],
        "drug2_num_side_effects": p2["num_side_effects"],
        "drug2_mean_rep_freq": p2["mean_rep_freq"],
        "drug2_total_A": p2["total_A"],
        "log_pair_avg_prr": log_pair_avg_prr,
        "log_pair_max_prr": log_pair_max_prr,
        "pair_snr": pair_snr,
        "drug1_snr": p1["snr"],
        "drug2_snr": p2["snr"],
        "relative_risk_ratio": relative_risk_ratio,
        "report_confidence": report_confidence,
        "effect_diversity_ratio": effect_diversity_ratio,
    })


def predict_interaction(drug_a_id: str, drug_b_id: str) -> Dict[str, Any]:
    models, imputer, feature_cols = load_artifacts(ARTIFACT_DIR)
    pf = load_pair_features(ARTIFACT_DIR)
    lookup = load_drug_lookup(ARTIFACT_DIR)

    if pf.empty:
        raise RuntimeError("pair_features.parquet not found")

    row = find_pair(drug_a_id, drug_b_id, pf)
    synthetic = row is None
    if row is None:
        row = _synthesize_pair_row(drug_a_id, drug_b_id, pf)

    row_df = row.to_frame().T
    feats = ensure_feature_columns(row_df, feature_cols)
    feats_imp = pd.DataFrame(imputer.transform(feats), columns=feature_cols)

    proba = {
        name: float(np.clip(model.predict_proba(feats_imp)[:, 1][0], 1e-6, 1 - 1e-6))
        for name, model in models.items()
    }

    # Simple hybrid = average of available classical probs
    hybrid_prob = float(np.mean(list(proba.values())))

    agreement = sum(p >= 0.5 for p in proba.values())
    total_models = len(proba)

    shap_info = compute_shap(models.get("xgboost"), feats_imp, feature_cols, sample_size=1)

    # JSON-safe cleanup for non-finite floats
    shap_info = {
        "global_top": [
            {"feature": x.get("feature", ""), "shap": _safe_float(x.get("shap"), 0.0)}
            for x in (shap_info.get("global_top") or [])
        ],
        "local_top": [
            {
                "feature": x.get("feature", ""),
                "shap": _safe_float(x.get("shap"), 0.0),
                "value": _safe_float(x.get("value"), 0.0),
            }
            for x in (shap_info.get("local_top") or [])
        ],
    }

    return {
        "drug_a_id": str(drug_a_id),
        "drug_b_id": str(drug_b_id),
        "drug_a_name": lookup.get(str(drug_a_id), str(drug_a_id)),
        "drug_b_name": lookup.get(str(drug_b_id), str(drug_b_id)),
        "proba": {k: _safe_float(v, 1e-6) for k, v in proba.items()},
        "hybrid_prob": _safe_float(hybrid_prob, 0.0),
        "risk": _risk_label(hybrid_prob),
        "confidence": _safe_float(float(agreement / total_models) if total_models else 0.0, 0.0),
        "agreement": f"{agreement}/{total_models}",
        "shap": shap_info,
        "synthetic_pair": synthetic,
        "features": {k: _safe_float(v, 0.0) for k, v in row.items() if k in feats.columns},
    }


def list_drugs() -> List[Dict[str, str]]:
    lookup = load_drug_lookup(ARTIFACT_DIR)
    if lookup:
        return [{"id": k, "name": v} for k, v in lookup.items()]

    pf = load_pair_features(ARTIFACT_DIR)
    if pf.empty:
        return []

    cols_ok = {"drug_1_rxnorm_id", "drug_2_rxnorm_id"}.issubset(set(pf.columns))
    if not cols_ok:
        return []

    ids = pd.concat([
        pf["drug_1_rxnorm_id"].astype(str),
        pf["drug_2_rxnorm_id"].astype(str),
    ], ignore_index=True).dropna().unique().tolist()

    return [{"id": str(i), "name": str(i)} for i in sorted(ids)]


def dataset_pairs(sample: Optional[int] = None) -> List[Dict[str, Any]]:
    pf = load_pair_features(ARTIFACT_DIR)
    lookup = load_drug_lookup(ARTIFACT_DIR)
    if pf.empty:
        return []
    if sample is not None:
        pf = pf.head(sample).copy()
    else:
        pf = pf.copy()
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
