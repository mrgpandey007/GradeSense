"""
train_without_grades.py
------------------------
Compares model performance WITH and WITHOUT the previous-term grades (G1, G2).

Why: G1/G2 carry almost all of the predictive power, so a model that uses
them mostly "copies the previous grade". An early-warning system should also
work at the start of a term, before G1/G2 exist. This script shows how much
performance drops (and which other factors matter) when G1/G2 are removed.

Input:
  data/processed/student_data_processed.csv

Output:
  outputs/reports/with_vs_without_grades.csv
  outputs/plots/11_with_vs_without_grades.png
  outputs/plots/12_feature_importance_without_grades.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, f1_score, recall_score,
)

PROCESSED_PATH = "data/processed/student_data_processed.csv"
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"
RANDOM_STATE = 42

sns.set_style("whitegrid")


def run_experiment(df, use_grades):
    """Train all 4 models on one feature set and return a dict of metrics."""
    drop_cols = ["G3", "pass_fail"] + ([] if use_grades else ["G1", "G2"])
    X = df.drop(columns=drop_cols)
    y_reg = df["G3"]
    y_clf = df["pass_fail"]

    # Same split for both targets and both versions (stratified on pass_fail)
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )

    results = {}

    # --- Regression ---
    lr = LinearRegression().fit(X_train, yr_train)
    rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE).fit(X_train, yr_train)
    for name, model in [("Linear Regression", lr), ("Random Forest Regressor", rfr)]:
        pred = model.predict(X_test)
        results[name] = {
            "R2": r2_score(yr_test, pred),
            "RMSE": np.sqrt(mean_squared_error(yr_test, pred)),
        }

    # --- Classification ---
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_train, yc_train)
    rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE).fit(X_train, yc_train)
    for name, model in [("Logistic Regression", logreg), ("Random Forest Classifier", rfc)]:
        pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(yc_test, pred),
            "F1": f1_score(yc_test, pred),
            # Recall of the FAIL class (label 0): how many at-risk students we catch
            "Fail Recall": recall_score(yc_test, pred, pos_label=0),
        }

    return results, rfr, X.columns


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")

    with_res, _, _ = run_experiment(df, use_grades=True)
    without_res, rfr_no_grades, cols_no_grades = run_experiment(df, use_grades=False)

    # ---- Build comparison table ----
    rows = []
    for model in with_res:
        for metric in with_res[model]:
            rows.append({
                "Model": model,
                "Metric": metric,
                "With G1/G2": round(with_res[model][metric], 4),
                "Without G1/G2": round(without_res[model][metric], 4),
            })
    table = pd.DataFrame(rows)
    table["Change"] = (table["Without G1/G2"] - table["With G1/G2"]).round(4)

    print("=" * 70)
    print("COMPARISON: WITH vs WITHOUT previous-term grades (G1, G2)")
    print("=" * 70)
    print(table.to_string(index=False))

    csv_path = f"{REPORTS_DIR}/with_vs_without_grades.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- Plot 1: R2 and Accuracy comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    r2 = table[table["Metric"] == "R2"]
    x = np.arange(len(r2))
    axes[0].bar(x - 0.2, r2["With G1/G2"], 0.4, label="With G1/G2", color="#1c5f8a")
    axes[0].bar(x + 0.2, r2["Without G1/G2"], 0.4, label="Without G1/G2", color="#e08a2c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(r2["Model"], rotation=10)
    axes[0].set_ylabel("R2 Score")
    axes[0].set_title("Regression (predicting G3): R2")
    axes[0].legend()

    acc = table[table["Metric"] == "Accuracy"]
    x = np.arange(len(acc))
    axes[1].bar(x - 0.2, acc["With G1/G2"], 0.4, label="With G1/G2", color="#1c5f8a")
    axes[1].bar(x + 0.2, acc["Without G1/G2"], 0.4, label="Without G1/G2", color="#e08a2c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(acc["Model"], rotation=10)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Classification (pass/fail): Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/11_with_vs_without_grades.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/11_with_vs_without_grades.png")

    # ---- Plot 2: feature importance when G1/G2 are removed ----
    imp = pd.Series(rfr_no_grades.feature_importances_, index=cols_no_grades)
    top = imp.sort_values(ascending=False).head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="Oranges_r", legend=False)
    plt.title("Top 10 Features WITHOUT G1/G2 (Random Forest Regressor)")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/12_feature_importance_without_grades.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/12_feature_importance_without_grades.png")

    print("\nTop features without G1/G2:")
    print(top.round(4).to_string())


if __name__ == "__main__":
    main()