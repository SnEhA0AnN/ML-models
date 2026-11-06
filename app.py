"""
Streamlit front end for ML Assignment 2 - Online Shoppers Purchasing Intention.
Loads the five pre-trained pipelines from model/ and scores an uploaded test CSV.
Nothing is retrained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
CLASS_NAMES = ["No purchase", "Purchase"]

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")


@st.cache_resource
def load_artifacts():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in info["files"].items()}
    return info, models


def to_binary(series):
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "false": 0, "0": 0})


def score_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    two_classes = y.nunique() == 2
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if two_classes else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_prob


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


info, models = load_artifacts()
FEATURES, TARGET = info["features"], info["target"]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Setup")
    chosen = st.selectbox("Choose a classifier", list(models.keys()))
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    st.download_button("Download sample test_data.csv",
                       data=(ROOT / "test_data.csv").read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    st.caption("The CSV must contain the 17 input columns and the `Revenue` column.")
    with st.expander("Tuned hyperparameters"):
        st.json(info["best_params"][chosen])
    st.divider()
    st.caption("Sneha Ann Samuel · 2026AJ05831  \nM.Tech AIML · Machine Learning, Assignment 2")

st.title("🛒 Online Shopper Purchase Predictor")
st.write("Predicts whether a browsing session on an e-commerce site ends in a purchase "
         "(UCI *Online Shoppers Purchasing Intention* dataset). Models were trained offline; "
         "this page only loads them and evaluates them on the CSV you upload.")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to see results. "
            "The sample file `test_data.csv` (the 20% hold-out split) can be downloaded there.")
    st.stop()

data = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"These required columns are missing from the file: {', '.join(missing)}")
    st.stop()

y_true = to_binary(data[TARGET])
if y_true.isna().any():
    st.error("`Revenue` should only contain True/False (or 1/0) values.")
    st.stop()
X = data[FEATURES]

c1, c2, c3 = st.columns(3)
c1.metric("Rows in file", f"{len(data):,}")
c2.metric("Actual purchases", f"{int(y_true.sum()):,}")
c3.metric("Purchase rate", f"{y_true.mean():.1%}")

tab_eval, tab_compare, tab_pred, tab_data = st.tabs(
    ["Selected model", "Compare all models", "Predictions", "Data preview"])

# ---------------- selected model ----------------
with tab_eval:
    metrics, y_pred, y_prob = score_model(models[chosen], X, y_true)
    st.subheader(f"{chosen} — evaluation metrics")
    cols = st.columns(3) + st.columns(3)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}", border=True)

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(y_true, y_pred, chosen))
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

# ---------------- all models ----------------
with tab_compare:
    rows, roc_data = [], {}
    for name, mdl in models.items():
        m, _, prob = score_model(mdl, X, y_true)
        rows.append({"ML Model Name": name, **m})
        roc_data[name] = prob
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.subheader("Metrics for all five models on the uploaded data")
    st.dataframe(table.style.format("{:.4f}").highlight_max(
        axis=0, props="background-color:#2e7d32; color:white; font-weight:bold"),
                 use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        table[["Accuracy", "AUC", "F1", "MCC"]].plot.bar(ax=ax, rot=20, width=0.75)
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Model comparison", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
    with g2:
        if y_true.nunique() == 2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            for name, prob in roc_data.items():
                fpr, tpr, _ = roc_curve(y_true, prob)
                ax.plot(fpr, tpr, label=f"{name} ({table.loc[name, 'AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves", fontsize=10)
            ax.legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("ROC curves need both classes in the uploaded file.")

# ---------------- predictions ----------------
with tab_pred:
    out = data.copy()
    out["Predicted"] = np.where(y_pred == 1, "Purchase", "No purchase")
    out["P(purchase)"] = y_prob.round(4)
    out["Correct"] = y_pred == y_true.values
    st.write(f"Row-level output of **{chosen}** "
             f"({out['Correct'].mean():.1%} of rows predicted correctly).")
    st.dataframe(out, use_container_width=True, height=420)
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{chosen.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

with tab_data:
    st.dataframe(data.head(50), use_container_width=True)
    st.caption(f"First 50 of {len(data):,} rows.")
