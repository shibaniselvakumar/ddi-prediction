#!/usr/bin/env python
"""
One-shot runner:
- Builds features from OFFSIDES + TWOSIDES
- Trains classical models (LR, RF+SMOTE, XGB)
- Runs SHAP (global + one local) on the tuned XGB
- Runs hybrid fusion (classical + NDD channel) with CV
- Runs a quick NDD mock test to ensure the neural path executes

Usage (from repo root):
    # Full run (dev-mode off): trains classical, SHAP, hybrid, NDD mock
    "c:/Users/shibs/OneDrive/Desktop/Projects/ML Project/NDD/.venv_win/Scripts/python.exe" run_all.py \
        --offsides data/OFFSIDES.csv.gz --twosides data/TWOSIDES.csv.gz \
        --data-dir NDD/DS1 --folds 3 --shap-samples 500 --ndd-mode mock

    # Dev/smoke (dev-mode on): skip retrain, reuse artifacts if present, tiny hybrid CV
    "c:/Users/shibs/OneDrive/Desktop/Projects/ML Project/NDD/.venv_win/Scripts/python.exe" run_all.py \
        --dev-mode --folds 2 --shap-samples 100 --ndd-mode mock

Notes:
- Assumes data files already exist in data/.
- SHAP runs on a small sample for speed.
- NDD channel defaults to mock MLP; provide a real Keras .h5 with --ndd-mode keras --ndd-model-path <path>.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pandas as pd
import shap
import joblib

import ddi_classical_pipeline as dcp
from hybrid_ddi_pipeline import HybridConfig, run_hybrid_cv

# Optional: quick neural mock test


def train_classical(offsides_path: Path, twosides_path: Path, output: Path, prr_threshold: float = 2.0, chunksize: int = 300_000, dev_mode: bool = False):
    # Dev mode: reuse existing artifacts if present, skip retrain
    needed = [output / "xgboost.pkl", output / "logreg.pkl", output / "random_forest.pkl", output / "imputer.pkl", output / "feature_cols.json"]
    if dev_mode and all(p.exists() for p in needed):
        print("\n[CLASSICAL] Dev mode: loading existing artifacts from", output)
        models = {
            "logreg": joblib.load(output / "logreg.pkl"),
            "random_forest": joblib.load(output / "random_forest.pkl"),
            "xgboost": joblib.load(output / "xgboost.pkl"),
        }
        imputer = joblib.load(output / "imputer.pkl")
        feature_cols = json.loads((output / "feature_cols.json").read_text())
        # Rebuild a small X_test from pair_features if available
        if (output / "pair_features.parquet").exists():
            pf = pd.read_parquet(output / "pair_features.parquet")
            # Ensure all feature columns exist and in the right order
            for col in feature_cols:
                if col not in pf.columns:
                    pf[col] = 0.0
            X_test = pf[feature_cols].head(300)
        else:
            X_test = None
        dataset_stub = SimpleNamespace(feature_cols=feature_cols)
        return models, dataset_stub, imputer, [], X_test, None
    elif dev_mode:
        print("[CLASSICAL] Dev mode requested but artifacts missing; falling back to full training.")

    print("\n[CLASSICAL] Building dataset...")
    dataset = dcp.build_dataset(dcp.DataPaths(str(offsides_path), str(twosides_path)), prr_threshold=prr_threshold, chunksize=chunksize)
    X_train, X_test, y_train, y_test, imputer = dcp.split_impute(dataset)

    print("[CLASSICAL] Training models (LR, RF+SMOTE, XGB)...")
    models, metrics, _ = dcp.train_all(dataset)

    print("[CLASSICAL] Metrics:")
    for m in metrics:
        print(f"  {m['name']:<18} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f}")

    print(f"[CLASSICAL] Saving artifacts to {output} ...")
    dcp.save_artifacts(models, imputer, dataset, str(output))
    return models, dataset, imputer, metrics, X_test, y_test


def run_shap_global_local(model, X: pd.DataFrame, feature_cols, sample_size: int = 500):
    if X is None or len(X) == 0:
        print("[SHAP] Skipped (no data)")
        return
    n = min(sample_size, len(X))
    X_sample = X.sample(n, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_feats = [(feature_cols[i], float(mean_abs[i])) for i in order[:10]]

    # Local explanation for top-scoring sample
    proba = model.predict_proba(X_sample)[:, 1]
    idx = int(np.argmax(proba))
    local_vals = shap_vals[idx]
    local_order = np.argsort(np.abs(local_vals))[::-1][:10]
    local_top = [(feature_cols[i], float(local_vals[i]), float(X_sample.iloc[idx, i])) for i in local_order]

    print("\n[SHAP] Global top-10 |SHAP| features:")
    for name, val in top_feats:
        print(f"  {name:<30} |SHAP|={val:.4f}")

    print("\n[SHAP] Local explanation for highest-risk sample:")
    for name, sv, fv in local_top:
        print(f"  {name:<30} SHAP={sv:.4f} value={fv:.4f}")


def run_hybrid(data_dir: Path, folds: int, ndd_mode: str, ndd_model_path: Path):
    print("\n[HYBRID] Running fusion CV...")
    cfg = HybridConfig(
        data_dir=str(data_dir),
        n_splits=folds,
        random_state=42,
        ndd_mode=ndd_mode,
        ndd_model_path=str(ndd_model_path) if ndd_model_path else None,
    )
    run_hybrid_cv(cfg)


def run_ndd_smoke():
    print("\n[NDD] Quick neural mock test (small subset)...")
    import importlib

    start_cwd = os.getcwd()
    try:
        # Importing quick_test executes its small mock training/eval.
        importlib.import_module("quick_test")
    finally:
        os.chdir(start_cwd)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run classical, SHAP, hybrid, and NDD smoke")
    parser.add_argument("--offsides", default=root / "data/OFFSIDES.csv.gz", type=Path)
    parser.add_argument("--twosides", default=root / "data/TWOSIDES.csv.gz", type=Path)
    parser.add_argument("--output", default=root / "model_artifacts", type=Path)
    parser.add_argument("--data-dir", default=root / "NDD/DS1", type=Path)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--shap-samples", type=int, default=500)
    parser.add_argument("--ndd-mode", choices=["mock", "keras", "none"], default="mock")
    parser.add_argument("--ndd-model-path", default=root / "ddi_model.h5", type=Path)
    parser.add_argument("--dev-mode", action="store_true", help="Reuse artifacts when present and run a smaller smoke.")
    return parser.parse_args()


def main():
    args = parse_args()

    # In dev mode, shrink workloads if user did not already set them smaller
    folds = min(args.folds, 2) if args.dev_mode else args.folds
    shap_samples = min(args.shap_samples, 150) if args.dev_mode else args.shap_samples
    chunksize = 120_000 if args.dev_mode else 300_000

    models, dataset, imputer, metrics, X_test, y_test = train_classical(
        args.offsides, args.twosides, args.output, chunksize=chunksize, dev_mode=args.dev_mode)

    # Impute dev-mode test slice to match training schema
    if X_test is not None and imputer is not None:
        X_test = pd.DataFrame(imputer.transform(X_test), columns=dataset.feature_cols)

    # Pick the best model for SHAP (use XGBoost if present)
    xgb_model = models.get("xgboost")
    if xgb_model is not None:
        run_shap_global_local(xgb_model, X_test, dataset.feature_cols, sample_size=shap_samples)
    else:
        print("[SHAP] Skipped (xgboost model not found)")

    run_hybrid(args.data_dir, folds, args.ndd_mode, args.ndd_model_path)
    run_ndd_smoke()

    print("\n[DONE] All stages executed.")


if __name__ == "__main__":
    main()
