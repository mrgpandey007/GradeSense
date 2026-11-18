"""
compare_models.py
------------------
Final model comparison. Adds three more models to the original four and
compares all of them with 5-fold cross-validation.

Regression (predict G3):
  Linear Regression, Random Forest, Gradient Boosting, SVR, KNN
Classification (predict pass/fail):
  Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN

Runs both feature sets: WITH G1/G2 and WITHOUT G1/G2.
Also records average training time per model.

Input:
  data/processed/student_data_processed.csv

Output:
  outputs/reports/model_comparison.csv
  outputs/plots/14_model_comparison_regression.png
  outputs/plots/15_model_comparison_classification.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import make_scorer, recall_score

PROCESSED_PATH = "data/processed/student_data_processed.csv"
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"
RS = 42
N_SPLITS = 5

sns.set_style("whitegrid")
fail_recall = make_scorer(recall_score, pos_label=0)

REG_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RS),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
    "SVR": SVR(kernel="rbf", C=5),
    "KNN": KNeighborsRegressor(n_neighbors=7),
}

CLF_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    "SVM": SVC(kernel="rbf", C=1, random_state=RS),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}


def run(df, use_grades):
    drop_cols = ["G3", "pass_fail"] + ([] if use_grades else ["G1", "G2"])
    X = df.drop(columns=drop_cols)
    y_reg, y_clf = df["G3"], df["pass_fail"]
    label = "With G1/G2" if use_grades else "Without G1/G2"

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    rows = []

    for name, model in REG_MODELS.items():
        cv = cross_validate(model, X, y_reg, cv=kf,
                            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})
        rows.append({"Task": "Regression", "Features": label, "Model": name,
                     "R2": cv["test_r2"].mean(), "RMSE": -cv["test_rmse"].mean(),
                     "Train Time (s)": cv["fit_time"].mean()})

    for name, model in CLF_MODELS.items():
        cv = cross_validate(model, X, y_clf, cv=skf,
                            scoring={"accuracy": "accuracy", "f1": "f1",
                                     "fail_recall": fail_recall})
        rows.append({"Task": "Classification", "Features": label, "Model": name,
                     "Accuracy": cv["test_accuracy"].mean(), "F1": cv["test_f1"].mean(),
                     "Fail Recall": cv["test_fail_recall"].mean(),
                     "Train Time (s)": cv["fit_time"].mean()})
    return rows


def plot_task(results, task, metric, filename, title):
    sub = results[results["Task"] == task]
    models = list(sub["Model"].unique())
    x = np.arange(len(models))
    colors = {"With G1/G2": "#1c5f8a", "Without G1/G2": "#e08a2c"}

    plt.figure(figsize=(10, 5.5))
    for i, feat in enumerate(["With G1/G2", "Without G1/G2"]):
        s = sub[sub["Features"] == feat].set_index("Model").loc[models]
        bars = plt.bar(x + (i - 0.5) * 0.4, s[metric], 0.4, label=feat, color=colors[feat])
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{b.get_height():.2f}", ha="center", fontsize=9)
    plt.xticks(x, models)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/{filename}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {df.shape[0]} rows. Comparing 5 models per task ({N_SPLITS}-fold CV)...\n")

    results = pd.DataFrame(run(df, True) + run(df, False)).round(4)

    reg = results[results["Task"] == "Regression"].dropna(axis=1, how="all")
    clf = results[results["Task"] == "Classification"].dropna(axis=1, how="all")

    print("=" * 78)
    print("REGRESSION (predicting G3)")
    print("=" * 78)
    print(reg.drop(columns="Task").to_string(index=False))
    print("\n" + "=" * 78)
    print("CLASSIFICATION (predicting pass/fail)")
    print("=" * 78)
    print(clf.drop(columns="Task").to_string(index=False))

    csv_path = f"{REPORTS_DIR}/model_comparison.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_task(results, "Regression", "R2", "14_model_comparison_regression.png",
              "Regression: R2 by model (5-fold CV)")
    plot_task(results, "Classification", "Accuracy", "15_model_comparison_classification.png",
              "Classification: Accuracy by model (5-fold CV)")

    # Best model summary (WITH grades and WITHOUT grades)
    print("\nBest models:")
    for feat in ["With G1/G2", "Without G1/G2"]:
        r = reg[reg["Features"] == feat].sort_values("R2", ascending=False).iloc[0]
        c = clf[clf["Features"] == feat].sort_values("Accuracy", ascending=False).iloc[0]
        print(f"  {feat}: Regression -> {r['Model']} (R2={r['R2']}), "
              f"Classification -> {c['Model']} (Accuracy={c['Accuracy']})")


if __name__ == "__main__":
    main()