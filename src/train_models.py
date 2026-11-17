"""
train_models.py
-----------------
Trains and saves:
  1. Regression models to predict G3 (final grade, 0-20):
       - Linear Regression
       - Random Forest Regressor
  2. Classification models to predict pass_fail (0/1):
       - Logistic Regression
       - Random Forest Classifier

Input:
  data/processed/student_data_processed.csv

Output:
  models/linear_regression.joblib
  models/random_forest_regressor.joblib
  models/logistic_regression.joblib
  models/random_forest_classifier.joblib
  models/feature_columns.joblib   (column order used at training time — needed for prediction later)
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

PROCESSED_PATH = "data/processed/student_data_processed.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_regression_data(df):
    # Predicting G3. We exclude G1 and G2 as well as pass_fail from the
    # feature set for one version (harder, more realistic task), but also
    # keep a version that includes G1/G2 since they're strong predictors
    # and commonly used in the original UCI-based studies.
    # Here we KEEP G1, G2 (common approach) but drop pass_fail (data leakage,
    # since pass_fail is directly derived from G3).
    X = df.drop(columns=["G3", "pass_fail"])
    y = df["G3"]
    return X, y


def prepare_classification_data(df):
    # Predicting pass_fail. Drop G3 (data leakage - pass_fail is derived
    # directly from G3) but keep G1, G2 as predictive features.
    X = df.drop(columns=["G3", "pass_fail"])
    y = df["pass_fail"]
    return X, y


def train_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n--- Regression: train={len(X_train)}, test={len(X_test)} ---")

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    joblib.dump(lr, f"{MODELS_DIR}/linear_regression.joblib")
    joblib.dump(rf, f"{MODELS_DIR}/random_forest_regressor.joblib")
    joblib.dump((X_test, y_test), f"{MODELS_DIR}/regression_test_set.joblib")

    print("Saved: linear_regression.joblib, random_forest_regressor.joblib")
    return lr, rf, X_test, y_test


def train_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n--- Classification: train={len(X_train)}, test={len(X_test)} ---")

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)

    rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    rfc.fit(X_train, y_train)

    joblib.dump(logreg, f"{MODELS_DIR}/logistic_regression.joblib")
    joblib.dump(rfc, f"{MODELS_DIR}/random_forest_classifier.joblib")
    joblib.dump((X_test, y_test), f"{MODELS_DIR}/classification_test_set.joblib")

    print("Saved: logistic_regression.joblib, random_forest_classifier.joblib")
    return logreg, rfc, X_test, y_test


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()

    # --- Regression task (predict G3) ---
    X_reg, y_reg = prepare_regression_data(df)
    joblib.dump(list(X_reg.columns), f"{MODELS_DIR}/feature_columns.joblib")
    train_regression_models(X_reg, y_reg)

    # --- Classification task (predict pass_fail) ---
    X_clf, y_clf = prepare_classification_data(df)
    train_classification_models(X_clf, y_clf)

    print("\nAll models trained and saved to models/")
    print("Next step: run evaluate.py to see accuracy, RMSE, R², confusion matrix, feature importance.")


if __name__ == "__main__":
    main()
