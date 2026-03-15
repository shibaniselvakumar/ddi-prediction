from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("DDI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Explainable Drug–Drug Interaction Prediction", layout="wide")


# ---------- Style ----------
st.markdown(
    """
    <style>
    .metric-card {border:1px solid #e5e7eb; padding:12px; border-radius:10px; background: #0b1021; color: #e8ecf2;}
    .section {padding:12px; border:1px solid #1f2937; border-radius:12px; background: #0f1629;}
    .risk-badge {padding:4px 10px; border-radius:8px; color:white; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_json(path: str, method: str = "get", params: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None):
    url = f"{API_URL}{path}"
    try:
        if method == "get":
            resp = requests.get(url, params=params, timeout=300)
        else:
            resp = requests.post(url, params=params, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:  # pragma: no cover
        st.error(f"Request failed ({exc.response.status_code}): {exc.response.text}")
    except Exception as exc:  # pragma: no cover
        st.error(f"Request failed: {exc}")
    return None


@st.cache_data(show_spinner=False)
def load_drugs() -> List[Dict[str, str]]:
    data = fetch_json("/drugs") or []
    return sorted(data, key=lambda x: x.get("name", ""))


@st.cache_data(show_spinner=False)
def load_pairs_sample(sample: int = 200) -> pd.DataFrame:
    data = fetch_json("/dataset/pairs", params={"sample": sample}) or []
    return pd.DataFrame(data)


def risk_badge(risk: str):
    color = {"Low": "#16a34a", "Moderate": "#f97316", "High": "#dc2626"}.get(risk, "#6b7280")
    st.markdown(f"<span class='risk-badge' style='background:{color}'>{risk}</span>", unsafe_allow_html=True)


def predict(drug_a: Dict[str, str], drug_b: Dict[str, str]):
    return fetch_json(
        "/predict",
        method="post",
        params={"drug_a_id": drug_a["id"], "drug_b_id": drug_b["id"]},
    )


def render_prediction_tab(drugs: List[Dict[str, str]]):
    st.subheader("Predict Interaction")
    names = [d["name"] for d in drugs]
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        a_name = st.selectbox("Drug A", names, index=0 if names else None)
    with col2:
        b_name = st.selectbox("Drug B", names, index=1 if len(names) > 1 else 0)
    run_btn = col3.button("Predict Interaction", use_container_width=True)

    if run_btn and a_name and b_name:
        drug_a = next(d for d in drugs if d["name"] == a_name)
        drug_b = next(d for d in drugs if d["name"] == b_name)
        with st.spinner("Running prediction..."):
            result = predict(drug_a, drug_b)
        if result:
            st.session_state["last_prediction"] = result
            render_prediction_result(result)
    elif "last_prediction" in st.session_state:
        render_prediction_result(st.session_state["last_prediction"])


def render_prediction_result(result: Dict[str, Any]):
    st.markdown(f"**Prediction for:** {result.get('drug_a_name')} + {result.get('drug_b_name')}")
    prob = result.get("hybrid_prob", 0.0)
    risk = result.get("risk", "n/a")
    confidence = result.get("confidence", 0.0)
    agreement = result.get("agreement", "-")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interaction Probability", f"{prob:.2f}")
    with c2:
        st.markdown("**Risk Level**")
        risk_badge(str(risk))
    c3.metric("Confidence", f"{confidence:.2f}")
    c4.metric("Model Agreement", str(agreement))

    st.markdown("**Per-model predictions**")
    proba = result.get("proba", {})
    if proba:
        df = pd.DataFrame([
            {"model": k, "probability": round(float(v), 3), "decision": "Yes" if v >= 0.5 else "No"}
            for k, v in proba.items()
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("model")["probability"])
    else:
        st.info("No model probabilities returned.")

    shap_info = result.get("shap", {})
    if shap_info:
        st.markdown("**Top contributing features (local)**")
        local = shap_info.get("local_top", [])
        if local:
            df_local = pd.DataFrame(local)
            st.bar_chart(df_local.set_index("feature")["shap"])
            st.dataframe(df_local, hide_index=True, use_container_width=True)


def render_explain_tab():
    st.subheader("Explain Prediction")
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction first.")
        return
    result = st.session_state["last_prediction"]
    shap_info = result.get("shap", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Local SHAP (top features)**")
        local = shap_info.get("local_top", [])
        if local:
            df_local = pd.DataFrame(local)
            st.dataframe(df_local, hide_index=True, use_container_width=True)
        else:
            st.info("No SHAP local info.")
    with col2:
        st.markdown("**Global SHAP (cached)**")
        global_feats = shap_info.get("global_top", [])
        if global_feats:
            df_g = pd.DataFrame(global_feats)
            st.bar_chart(df_g.set_index("feature")["shap"])
        else:
            st.info("No SHAP global info.")


def render_model_comparison_tab():
    st.subheader("Model Comparison")
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction first.")
        return
    proba = st.session_state["last_prediction"].get("proba", {})
    if not proba:
        st.info("No per-model probabilities available.")
        return
    df = pd.DataFrame([
        {"Model": k, "Probability": round(float(v), 3), "Decision": "Interaction" if v >= 0.5 else "No"}
        for k, v in proba.items()
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Model")["Probability"])


def render_dataset_tab():
    st.subheader("Dataset Explorer")
    df = load_pairs_sample()
    if df.empty:
        st.info("No dataset sample available.")
        return
    search = st.text_input("Search drug name")
    if search:
        mask = df["drug1_name"].str.contains(search, case=False, na=False) | df["drug2_name"].str.contains(search, case=False, na=False)
        df = df[mask]
    st.dataframe(df, use_container_width=True)


def render_system_tab():
    st.subheader("System Metrics")
    latest = fetch_json("/runs/latest") or {}
    if latest:
        st.json(latest)
    artifacts = fetch_json("/artifacts") or []
    st.markdown("**Artifacts**")
    if artifacts:
        st.dataframe(pd.DataFrame(artifacts), use_container_width=True)
    preds = fetch_json("/predictions/sample") or []
    st.markdown("**Sample predictions (XGB cached)**")
    if preds:
        st.dataframe(pd.DataFrame(preds), use_container_width=True)


# -------------- Layout --------------
st.title("Explainable Drug–Drug Interaction Prediction System")

drugs = load_drugs()
tabs = st.tabs([
    "Predict Interaction",
    "Explain Prediction",
    "Model Comparison",
    "Dataset Explorer",
    "System Metrics",
])

with tabs[0]:
    render_prediction_tab(drugs)
with tabs[1]:
    render_explain_tab()
with tabs[2]:
    render_model_comparison_tab()
with tabs[3]:
    render_dataset_tab()
with tabs[4]:
    render_system_tab()
