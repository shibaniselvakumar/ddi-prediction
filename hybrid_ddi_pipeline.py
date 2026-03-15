#!/usr/bin/env python
"""
Hybrid DDI pipeline: Classical ML + NDD score fusion.

Goal
----
Keep your friend's classical pipeline and your NDD model separate,
then combine both predictions in a calibrated meta-model.

Why this is robust
------------------
- If one model overfits, the other can stabilize final prediction.
- Fusion can improve PR-AUC on imbalanced DDI data.
- Components remain modular and easy to debug.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


@dataclass
class HybridConfig:
    data_dir: str
    n_splits: int = 5
    random_state: int = 42
    ndd_mode: str = "mock"  # mock | keras | none
    ndd_model_path: Optional[str] = None


class KerasNDDWrapper:
    """Loads a Keras .h5 model and returns class-1 probabilities."""

    def __init__(self, model_path: str):
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow is required for --ndd-mode keras. "
                "Install with: pip install tensorflow"
            ) from exc

        self._tf = tf
        self.model = tf.keras.models.load_model(model_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = self.model.predict(X, verbose=0)
        out = np.asarray(out)

        if out.ndim == 1:
            p1 = out
        elif out.shape[1] == 1:
            p1 = out[:, 0]
        else:
            p1 = out[:, 1]

        return np.clip(p1.astype(float), 0.0, 1.0)


def load_ds1_pair_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pairwise features from DS1 data.

    X(i,j) = concat(feature(i), feature(j))
    y(i,j) = interaction[i,j]
    """
    feat_path = os.path.join(data_dir, "IntegratedDS1.txt")
    y_path = os.path.join(data_dir, "drug_drug_matrix.csv")

    drug_features = np.loadtxt(feat_path, dtype=float, delimiter=",")
    interaction = np.loadtxt(y_path, dtype=int, delimiter=",")

    n = interaction.shape[0]
    X = np.empty((n * n, drug_features.shape[1] * 2), dtype=float)
    y = np.empty(n * n, dtype=int)

    row = 0
    for i in range(n):
        fi = drug_features[i]
        for j in range(n):
            fj = drug_features[j]
            X[row] = np.concatenate([fi, fj])
            y[row] = interaction[i, j]
            row += 1

    return X, y


def make_classical_model(seed: int):
    # Strong tabular baseline; calibrated probabilities help fusion.
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def make_mock_ndd_model(seed: int):
    # Fast surrogate channel to test hybrid plumbing when Keras model not available.
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=20,
        random_state=seed,
    )


def evaluate_fold(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_hat = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
    }


def run_hybrid_cv(cfg: HybridConfig):
    X, y = load_ds1_pair_data(cfg.data_dir)
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    scores = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Channel 1: classical model
        classical = make_classical_model(cfg.random_state + fold)
        classical.fit(X_tr, y_tr)
        p_classical_tr = classical.predict_proba(X_tr)[:, 1]
        p_classical_te = classical.predict_proba(X_te)[:, 1]

        # Channel 2: NDD score
        if cfg.ndd_mode == "keras":
            if not cfg.ndd_model_path:
                raise ValueError("--ndd-model-path is required when --ndd-mode keras")
            ndd = KerasNDDWrapper(cfg.ndd_model_path)
            p_ndd_tr = ndd.predict_proba(X_tr)
            p_ndd_te = ndd.predict_proba(X_te)

        elif cfg.ndd_mode == "mock":
            ndd_mock = make_mock_ndd_model(cfg.random_state + 100 + fold)
            ndd_mock.fit(X_tr, y_tr)
            p_ndd_tr = ndd_mock.predict_proba(X_tr)[:, 1]
            p_ndd_te = ndd_mock.predict_proba(X_te)[:, 1]

        elif cfg.ndd_mode == "none":
            p_ndd_tr = np.zeros_like(p_classical_tr)
            p_ndd_te = np.zeros_like(p_classical_te)

        else:
            raise ValueError(f"Unsupported ndd_mode: {cfg.ndd_mode}")

        # Meta fusion (late fusion)
        X_meta_tr = np.column_stack([p_classical_tr, p_ndd_tr])
        X_meta_te = np.column_stack([p_classical_te, p_ndd_te])

        meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.random_state)
        meta.fit(X_meta_tr, y_tr)
        p_final = meta.predict_proba(X_meta_te)[:, 1]

        fold_metrics = evaluate_fold(y_te, p_final)
        scores.append(fold_metrics)

        print(
            f"Fold {fold}/{cfg.n_splits} | "
            f"ROC-AUC={fold_metrics['roc_auc']:.4f} "
            f"PR-AUC={fold_metrics['pr_auc']:.4f} "
            f"F1={fold_metrics['f1']:.4f}"
        )

    mean_metrics = {k: float(np.mean([s[k] for s in scores])) for k in scores[0].keys()}

    print("\n=== Hybrid CV Mean Metrics ===")
    for k, v in mean_metrics.items():
        print(f"{k:>10}: {v:.4f}")

    return {"folds": scores, "mean": mean_metrics}


def parse_args() -> HybridConfig:
    root = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Hybrid DDI pipeline (Classical + NDD fusion)")
    parser.add_argument("--data-dir", default=os.path.join(root, "NDD", "DS1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ndd-mode", choices=["mock", "keras", "none"], default="mock")
    parser.add_argument("--ndd-model-path", default=os.path.join(root, "ddi_model.h5"))

    args = parser.parse_args()

    return HybridConfig(
        data_dir=args.data_dir,
        n_splits=args.folds,
        random_state=args.seed,
        ndd_mode=args.ndd_mode,
        ndd_model_path=args.ndd_model_path,
    )


if __name__ == "__main__":
    config = parse_args()
    run_hybrid_cv(config)
