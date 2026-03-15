#!/usr/bin/env python
"""
Clean, reusable DDI classical pipeline extracted from ddi_pipeline.py.
- Builds features from NSIDES OffSIDES + TwoSIDES
- Trains Logistic Regression, RandomForest+SMOTE, XGBoost
- Exposes utilities for saving artifacts and scoring pairwise risks

Note: this does not auto-download NSIDES. Place OFFSIDES.csv.gz and
TWOSIDES.csv.gz under a data directory (default: ./data).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: List[str] = [
    # Pair-level raw features
    "pair_avg_prr",
    "pair_max_prr",
    "pair_prr_std",
    "pair_num_effects",
    "pair_A_sum",
    "pair_B_sum",
    "pair_avg_prr_error",
    "pair_avg_rep_freq",
    # Drug 1 individual features
    "drug1_avg_prr",
    "drug1_max_prr",
    "drug1_prr_std",
    "drug1_num_side_effects",
    "drug1_mean_rep_freq",
    "drug1_total_A",
    # Drug 2 individual features
    "drug2_avg_prr",
    "drug2_max_prr",
    "drug2_prr_std",
    "drug2_num_side_effects",
    "drug2_mean_rep_freq",
    "drug2_total_A",
    # Derived statistics
    "log_pair_avg_prr",
    "log_pair_max_prr",
    "pair_snr",
    "drug1_snr",
    "drug2_snr",
    "relative_risk_ratio",
    "report_confidence",
    "effect_diversity_ratio",
]


@dataclass
class DataPaths:
    offsides_path: str
    twosides_path: str


@dataclass
class Dataset:
    X_raw: pd.DataFrame
    y: pd.Series
    feature_cols: List[str]
    drug_features: pd.DataFrame
    pair_features: pd.DataFrame
    drug_name_lookup: Dict[int, str]


def load_offsides(path: str) -> pd.DataFrame:
    # Read with low_memory=False to avoid mixed dtypes, then coerce numeric cols.
    offsides = pd.read_csv(path, compression="gzip", low_memory=False)
    numeric_cols = ["A", "B", "C", "D", "PRR", "PRR_error", "mean_reporting_frequency"]
    for col in numeric_cols:
        if col in offsides.columns:
            offsides[col] = pd.to_numeric(offsides[col], errors="coerce")

    offsides = offsides[
        (offsides["A"] >= 2)
        & (offsides["PRR_error"].notna())
        & (offsides["PRR"] > 0)
    ].copy()
    return offsides


def build_drug_features(offsides: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    df = offsides.groupby("drug_rxnorn_id").agg(
        drug_avg_prr=("PRR", "mean"),
        drug_max_prr=("PRR", "max"),
        drug_prr_std=("PRR", "std"),
        drug_num_side_effects=("condition_meddra_id", "nunique"),
        drug_mean_rep_freq=("mean_reporting_frequency", "mean"),
        drug_total_A=("A", "sum"),
        drug_avg_prr_error=("PRR_error", "mean"),
    ).reset_index()
    df["drug_prr_std"] = df["drug_prr_std"].fillna(0)

    name_lookup = (
        offsides[["drug_rxnorn_id", "drug_concept_name"]]
        .drop_duplicates()
        .set_index("drug_rxnorn_id")["drug_concept_name"]
        .to_dict()
    )
    df = df.rename(columns={"drug_rxnorn_id": "drug_rxnorm_id"})
    return df, name_lookup


def _infer_twosides_column_map(header: pd.Index) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in header:
        c = col.strip().lower()
        if "rxnorn" in c and "1" in c:
            mapping[col] = "drug_1_rxnorm_id"
        elif "rxnorm" in c and "2" in c:
            mapping[col] = "drug_2_rxnorm_id"
        elif "concept_name" in c and "1" in c:
            mapping[col] = "drug_1_concept_name"
        elif "concept_name" in c and ("2" in c or "3" in c):
            mapping[col] = "drug_2_concept_name"
        elif "meddra" in c:
            mapping[col] = "condition_meddra_id"
        elif "concept_name" in c:
            mapping[col] = "condition_concept_name"
        elif c == "a":
            mapping[col] = "A"
        elif c == "b":
            mapping[col] = "B"
        elif c == "c":
            mapping[col] = "C"
        elif c == "d":
            mapping[col] = "D"
        elif c == "prr" and "error" not in c:
            mapping[col] = "PRR"
        elif "prr_error" in c or "prr error" in c:
            mapping[col] = "PRR_error"
        elif "reporting_frequency" in c or "reporting freq" in c:
            mapping[col] = "mean_reporting_frequency"
    return mapping


def build_pair_features(path: str, chunksize: int = 300_000) -> pd.DataFrame:
    header = pd.read_csv(path, compression="gzip", nrows=0)
    col_map = _infer_twosides_column_map(header.columns)
    num_cols = ["A", "B", "C", "D", "PRR", "PRR_error", "mean_reporting_frequency"]

    pair_chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=chunksize,
        usecols=list(col_map.keys()),
    ):
        chunk = chunk.rename(columns=col_map)
        for col in num_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk[(chunk["A"] >= 5) & (chunk["PRR_error"].notna()) & (chunk["PRR"] > 0)]
        chunk = chunk[chunk["condition_concept_name"].str.lower() != "unevaluable event"]
        if len(chunk) == 0:
            continue

        top_fx_idx = chunk.groupby(["drug_1_rxnorm_id", "drug_2_rxnorm_id"])["PRR"].idxmax()
        top_fx = chunk.loc[top_fx_idx, ["drug_1_rxnorm_id", "drug_2_rxnorm_id", "condition_concept_name"]]
        top_fx = top_fx.rename(columns={"condition_concept_name": "top_effect"})

        agg = chunk.groupby(["drug_1_rxnorm_id", "drug_2_rxnorm_id"]).agg(
            pair_avg_prr=("PRR", "mean"),
            pair_max_prr=("PRR", "max"),
            pair_prr_std=("PRR", "std"),
            pair_num_effects=("condition_meddra_id", "nunique"),
            pair_A_sum=("A", "sum"),
            pair_B_sum=("B", "sum"),
            pair_C_sum=("C", "sum"),
            pair_D_sum=("D", "sum"),
            pair_avg_prr_error=("PRR_error", "mean"),
            pair_avg_rep_freq=("mean_reporting_frequency", "mean"),
        ).reset_index()

        agg = agg.merge(top_fx, on=["drug_1_rxnorm_id", "drug_2_rxnorm_id"], how="left")
        pair_chunks.append(agg)

    raw = pd.concat(pair_chunks, ignore_index=True)
    pair = raw.groupby(["drug_1_rxnorm_id", "drug_2_rxnorm_id"]).agg({
        "pair_avg_prr": "mean",
        "pair_max_prr": "max",
        "pair_prr_std": "mean",
        "pair_num_effects": "sum",
        "pair_A_sum": "sum",
        "pair_B_sum": "sum",
        "pair_C_sum": "sum",
        "pair_D_sum": "sum",
        "pair_avg_prr_error": "mean",
        "pair_avg_rep_freq": "mean",
        "top_effect": "first",
    }).reset_index()

    pair["pair_prr_std"] = pair["pair_prr_std"].fillna(0)
    return pair


def _merge_features(pair_features: pd.DataFrame, drug_features: pd.DataFrame) -> pd.DataFrame:
    d1 = drug_features.add_prefix("drug1_").rename(columns={"drug1_drug_rxnorm_id": "drug_1_rxnorm_id"})
    d2 = drug_features.add_prefix("drug2_").rename(columns={"drug2_drug_rxnorm_id": "drug_2_rxnorm_id"})
    df = pair_features.merge(d1, on="drug_1_rxnorm_id", how="left").merge(d2, on="drug_2_rxnorm_id", how="left")
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df["log_pair_avg_prr"] = np.log(df["pair_avg_prr"] + eps)
    df["log_pair_max_prr"] = np.log(df["pair_max_prr"] + eps)
    df["pair_snr"] = df["pair_avg_prr"] / (df["pair_avg_prr_error"] + eps)
    df["drug1_snr"] = df["drug1_drug_avg_prr"] / (df["drug1_drug_avg_prr_error"] + eps)
    df["drug2_snr"] = df["drug2_drug_avg_prr"] / (df["drug2_drug_avg_prr_error"] + eps)
    df["relative_risk_ratio"] = df["pair_avg_prr"] / (df["drug1_drug_avg_prr"] * df["drug2_drug_avg_prr"] + eps)
    df["report_confidence"] = df["pair_A_sum"] / (df["pair_A_sum"] + df["pair_B_sum"] + df.get("pair_C_sum", 0) + df.get("pair_D_sum", 0) + eps)
    df["effect_diversity_ratio"] = df["pair_num_effects"] / (df["drug1_drug_num_side_effects"] + df["drug2_drug_num_side_effects"] + eps)
    return df


def build_dataset(paths: DataPaths, prr_threshold: float = 2.0, chunksize: int = 300_000) -> Dataset:
    offs = load_offsides(paths.offsides_path)
    drug_features, lookup = build_drug_features(offs)
    pair_features = build_pair_features(paths.twosides_path, chunksize=chunksize)

    df = _merge_features(pair_features, drug_features)
    df = _add_derived_features(df)
    df = df.rename(columns={
        "drug1_drug_avg_prr": "drug1_avg_prr",
        "drug2_drug_avg_prr": "drug2_avg_prr",
        "drug1_drug_max_prr": "drug1_max_prr",
        "drug2_drug_max_prr": "drug2_max_prr",
        "drug1_drug_prr_std": "drug1_prr_std",
        "drug2_drug_prr_std": "drug2_prr_std",
        "drug1_drug_num_side_effects": "drug1_num_side_effects",
        "drug2_drug_num_side_effects": "drug2_num_side_effects",
        "drug1_drug_mean_rep_freq": "drug1_mean_rep_freq",
        "drug2_drug_mean_rep_freq": "drug2_mean_rep_freq",
        "drug1_drug_total_A": "drug1_total_A",
        "drug2_drug_total_A": "drug2_total_A",
        "drug1_drug_avg_prr_error": "drug1_avg_prr_error",
        "drug2_drug_avg_prr_error": "drug2_avg_prr_error",
    })

    y = (df["pair_max_prr"] >= prr_threshold).astype(int)
    X_raw = df[FEATURE_COLS].copy()
    return Dataset(X_raw=X_raw, y=y, feature_cols=FEATURE_COLS, drug_features=drug_features, pair_features=pair_features, drug_name_lookup=lookup)


def split_impute(dataset: Dataset, test_size: float = 0.2, seed: int = 42):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(dataset.X_raw, dataset.y, test_size=test_size, stratify=dataset.y, random_state=seed)
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=dataset.feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=dataset.feature_cols)
    return X_train, X_test, y_train, y_test, imputer


def train_logreg(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_rf_smote(X_train, y_train):
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train, y_train, y_test):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_train, y_train)], verbose=False)
    return model


def evaluate_model(name: str, model, X_test, y_test) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "name": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
    }


def train_all(dataset: Dataset, seed: int = 42):
    X_train, X_test, y_train, y_test, imputer = split_impute(dataset, seed=seed)

    models = {}
    metrics = []

    lr = train_logreg(X_train, y_train)
    models["logreg"] = lr
    metrics.append(evaluate_model("LogisticRegression", lr, X_test, y_test))

    rf = train_rf_smote(X_train, y_train)
    models["random_forest"] = rf
    metrics.append(evaluate_model("RandomForest", rf, X_test, y_test))

    xgb_model = train_xgb(X_train, y_train, y_test)
    models["xgboost"] = xgb_model
    metrics.append(evaluate_model("XGBoost", xgb_model, X_test, y_test))

    return models, metrics, imputer


def save_artifacts(models, imputer, dataset: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    import joblib

    for name, model in models.items():
        joblib.dump(model, os.path.join(out_dir, f"{name}.pkl"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))
    # Ensure ID columns are strings to avoid mixed-type parquet errors
    df_drug = dataset.drug_features.copy()
    if "drug_rxnorm_id" in df_drug.columns:
        df_drug["drug_rxnorm_id"] = df_drug["drug_rxnorm_id"].astype(str)

    df_pair = dataset.pair_features.copy()
    for c in ["drug_1_rxnorm_id", "drug_2_rxnorm_id"]:
        if c in df_pair.columns:
            df_pair[c] = df_pair[c].astype(str)

    df_drug.to_parquet(os.path.join(out_dir, "drug_features.parquet"), index=False)
    df_pair.to_parquet(os.path.join(out_dir, "pair_features.parquet"), index=False)
    with open(os.path.join(out_dir, "feature_cols.json"), "w") as f:
        json.dump(dataset.feature_cols, f)
    with open(os.path.join(out_dir, "drug_name_lookup.json"), "w") as f:
        json.dump({str(k): v for k, v in dataset.drug_name_lookup.items()}, f)


def explain_with_shap(model, X_sample: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    return shap_vals


def parse_args():
    root = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(root, "data")
    parser = argparse.ArgumentParser(description="Train classical DDI models from NSIDES")
    parser.add_argument("--offsides", default=os.path.join(default_data, "OFFSIDES.csv.gz"))
    parser.add_argument("--twosides", default=os.path.join(default_data, "TWOSIDES.csv.gz"))
    parser.add_argument("--prr-threshold", type=float, default=2.0)
    parser.add_argument("--chunksize", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(root, "model_artifacts"))
    return parser.parse_args()


def main():
    args = parse_args()
    paths = DataPaths(offsides_path=args.offsides, twosides_path=args.twosides)
    dataset = build_dataset(paths, prr_threshold=args.prr_threshold, chunksize=args.chunksize)
    models, metrics, imputer = train_all(dataset, seed=args.seed)
    save_artifacts(models, imputer, dataset, args.output)

    print("\n=== Metrics ===")
    for m in metrics:
        print(f"{m['name']:<18} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f}")

    print(f"Artifacts written to: {args.output}")


if __name__ == "__main__":
    main()
