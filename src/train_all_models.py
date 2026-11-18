"""
train_all_models.py
--------------------
Trains ALL models for BOTH feature sets and saves them in one bundle that
the Streamlit app loads.

Feature sets:
  - "with"    : uses G1 and G2 (previous-term grades)
  - "without" : no G1/G2 (early-warning setting)

Models per feature set:
  Regression     : Linear Regression, Random Forest, Gradient Boosting, SVR, KNN
  Classification : Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN

Each model is trained on an 80% split and its test-set metrics are stored,
so the app can show honest performance numbers next to every prediction.

Input:
  data/processed/student_data_processed.csv

Output:
  models/model_bundle.joblib
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, f1_score, recall_score,
)

PROCESSED_PATH = "data/processed/student_data_processed.csv"
MODELS_DIR = "models"
RS = 42


def reg_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RS),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
        "SVR": SVR(kernel="rbf", C=5),
        "KNN": KNeighborsRegressor(n_neighbors=7),
    }


def clf_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
        "SVM": SVC(kernel="rbf", C=1, probability=True, random_state=RS),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_feature_set(df, use_grades):
    drop_cols = ["G3", "pass_fail"] + ([] if use_grades else ["G1", "G2"])
    X = df.drop(columns=drop_cols)
    y_reg, y_clf = df["G3"], df["pass_fail"]

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RS, stratify=y_clf
    )

    bundle = {"features": list(X.columns), "regression": {}, "classification": {}}

    for name, model in reg_models().items():
        model.fit(X_train, yr_train)
        pred = model.predict(X_test)
        bundle["regression"][name] = {
            "model": model,
            "metrics": {
                "R2": float(r2_score(yr_test, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(yr_test, pred))),
            },
        }

    for name, model in clf_models().items():
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)
        bundle["classification"][name] = {
            "model": model,
            "metrics": {
                "Accuracy": float(accuracy_score(yc_test, pred)),
                "F1": float(f1_score(yc_test, pred)),
                "Fail Recall": float(recall_score(yc_test, pred, pos_label=0)),
            },
        }

    return bundle


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")

    full_bundle = {
        "with": train_feature_set(df, use_grades=True),
        "without": train_feature_set(df, use_grades=False),
    }

    for key, label in [("with", "WITH G1/G2"), ("without", "WITHOUT G1/G2")]:
        print(f"=== {label} ({len(full_bundle[key]['features'])} features) ===")
        for name, item in full_bundle[key]["regression"].items():
            m = item["metrics"]
            print(f"  [Regression]     {name:<20} R2={m['R2']:.3f}  RMSE={m['RMSE']:.2f}")
        for name, item in full_bundle[key]["classification"].items():
            m = item["metrics"]
            print(f"  [Classification] {name:<20} Acc={m['Accuracy']:.3f}  F1={m['F1']:.3f}  FailRecall={m['Fail Recall']:.3f}")
        print()

    out_path = f"{MODELS_DIR}/model_bundle.joblib"
    joblib.dump(full_bundle, out_path, compress=3)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()