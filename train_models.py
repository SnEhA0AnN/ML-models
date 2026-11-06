"""
ML Assignment 2 - Online Shoppers Purchasing Intention (UCI id 468)
Sneha Ann Samuel | 2026AJ05831

Trains Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM
on the same 80/20 stratified split, evaluates them on the held-out 20% and
saves every fitted pipeline (preprocessing + classifier) into model/.

Run:  python train_models.py --data data/online_shoppers_intention.csv
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
# harmless joblib message on Windows machines without wmic
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "Revenue"

# page counts / durations / PageValues are heavily right-skewed -> log1p first
SKEWED_COLS = ["Administrative", "Administrative_Duration", "Informational",
               "Informational_Duration", "ProductRelated",
               "ProductRelated_Duration", "PageValues"]
RATE_COLS = ["BounceRates", "ExitRates", "SpecialDay"]
# integer coded columns in the source are really categories (ids), not magnitudes
CAT_COLS = ["Month", "VisitorType", "Weekend", "OperatingSystems",
            "Browser", "Region", "TrafficType"]
FEATURES = SKEWED_COLS + RATE_COLS + CAT_COLS

MODEL_DIR = Path(__file__).parent / "model"
FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def to_binary(series):
    """Revenue comes as True/False; also accept 1/0 or 'TRUE'/'FALSE' text."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "false": 0, "0": 0})


def build_preprocessor():
    skew_pipe = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())])
    return ColumnTransformer([
        ("skewed", skew_pipe, SKEWED_COLS),
        ("rates", StandardScaler(), RATE_COLS),
        ("cats", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])


def search_spaces():
    # (estimator, grid) - grids kept small so the full run finishes in a few minutes
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
             "clf__class_weight": [None, "balanced"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [3, 4, 5, 6, 8, 12, None],
             "clf__min_samples_leaf": [1, 10, 30],
             "clf__class_weight": [None, "balanced"]}),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [11, 21, 31, 41, 51, 71], "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0]}),
        # probability=False during the search (faster), switched on for the final fit
        "Support Vector Machine": (
            SVC(kernel="rbf", random_state=SEED),
            {"clf__C": [0.02, 0.05, 0.1, 0.25, 0.5, 1, 5], "clf__class_weight": [None, "balanced"]}),
    }


def evaluate(pipe, X, y):
    y_pred = pipe.predict(X)
    y_score = pipe.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main(data_path):
    raw = pd.read_csv(data_path)
    print(f"loaded {raw.shape[0]} rows x {raw.shape[1]} cols")
    raw = raw.drop_duplicates().reset_index(drop=True)
    print(f"after removing duplicates: {raw.shape[0]} rows")

    X = raw[FEATURES]
    y = to_binary(raw[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train {len(X_tr)} | test {len(X_te)} | purchase rate {y.mean():.3f}")

    # held-out test split, in the original column format, for the Streamlit app
    test_df = raw.loc[X_te.index, FEATURES + [TARGET]]
    test_df.to_csv(Path(__file__).parent / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for name, (estimator, grid) in search_spaces().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        final = gs.best_estimator_
        if name == "Support Vector Machine":
            final = clone(final).set_params(clf__probability=True).fit(X_tr, y_tr)

        scores = evaluate(final, X_te, y_te)
        rows.append({"ML Model Name": name, **scores})
        best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        joblib.dump(final, MODEL_DIR / FILE_NAMES[name])
        print(f"{name:<24} cv-F1={gs.best_score_:.4f}  test: "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    results = pd.DataFrame(rows).round(4)
    results.to_csv(MODEL_DIR / "test_metrics.csv", index=False)
    with open(MODEL_DIR / "model_info.json", "w") as fh:
        json.dump({"target": TARGET, "features": FEATURES, "categorical": CAT_COLS,
                   "files": FILE_NAMES, "best_params": best_params,
                   "train_rows": len(X_tr), "test_rows": len(X_te), "seed": SEED},
                  fh, indent=2, default=str)
    print("\n", results.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/online_shoppers_intention.csv")
    main(ap.parse_args().data)
