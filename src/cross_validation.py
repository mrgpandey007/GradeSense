"""
cross_validation.py
--------------------
Evaluates all 4 models with 5-fold cross-validation instead of a single
80-20 split. With only ~395 records, one split can be lucky or unlucky;
averaging over 5 folds gives a more reliable estimate, and the standard
deviation shows how stable each model is.

Runs two feature sets:
  - WITH G1/G2 (previous-term grades)
  - WITHOUT G1/G2 (early-warning setting)

Input:
  data/processed/student_data_processed.csv

Output:
  outputs/reports/cross_validation_results.csv
  outputs/plots/13_cross_validation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score

PROCESSED_PATH = "data/processed/student_data_processed.csv"
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"
RANDOM_STATE = 42
N_SPLITS = 5

sns.set_style("whitegrid")

# Recall of the FAIL class (label 0) = share of at-risk students we catch
fail_recall = make_scorer(recall_score, pos_label=0)


def evaluate_feature_set(df, use_grades):
    drop_cols = ["G3", "pass_fail"] + ([] if use_grades else ["G1", "G2"])
    X = df.drop(columns=drop_cols)
    y_reg = df["G3"]
    y_clf = df["pass_fail"]
    label = "With G1/G2" if use_grades else "Without G1/G2"

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    # --- Regression models ---
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    }
    for name, model in reg_models.items():
        cv = cross_validate(
            model, X, y_reg, cv=kf,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        )
        rows.append({
            "Features": label, "Model": name, "Task": "Regression",
            "Metric": "R2",
            "Mean": cv["test_r2"].mean(), "Std": cv["test_r2"].std(),
        })
        rows.append({
            "Features": label, "Model": name, "Task": "Regression",
            "Metric": "RMSE",
            "Mean": -cv["test_rmse"].mean(), "Std": cv["test_rmse"].std(),
        })

    # --- Classification models ---
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    }
    for name, model in clf_models.items():
        cv = cross_validate(
            model, X, y_clf, cv=skf,
            scoring={"accuracy": "accuracy", "f1": "f1", "fail_recall": fail_recall},
        )
        for metric, key in [("Accuracy", "test_accuracy"), ("F1", "test_f1"),
                            ("Fail Recall", "test_fail_recall")]:
            rows.append({
                "Features": label, "Model": name, "Task": "Classification",
                "Metric": metric,
                "Mean": cv[key].mean(), "Std": cv[key].std(),
            })

    return rows


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {df.shape[0]} rows. Running {N_SPLITS}-fold cross-validation...\n")

    rows = evaluate_feature_set(df, use_grades=True) + evaluate_feature_set(df, use_grades=False)
    results = pd.DataFrame(rows)
    results["Mean"] = results["Mean"].round(4)
    results["Std"] = results["Std"].round(4)
    results["Mean +/- Std"] = results["Mean"].astype(str) + " +/- " + results["Std"].astype(str)

    print("=" * 78)
    print(f"{N_SPLITS}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 78)
    print(results[["Features", "Model", "Metric", "Mean +/- Std"]].to_string(index=False))

    csv_path = f"{REPORTS_DIR}/cross_validation_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- Plot: R2 (regression) and Accuracy (classification) with error bars ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"With G1/G2": "#1c5f8a", "Without G1/G2": "#e08a2c"}

    for ax, metric, title in [(axes[0], "R2", "Regression: R2 (5-fold CV)"),
                              (axes[1], "Accuracy", "Classification: Accuracy (5-fold CV)")]:
        sub = results[results["Metric"] == metric]
        models = sub["Model"].unique()
        x = np.arange(len(models))
        for i, feat in enumerate(["With G1/G2", "Without G1/G2"]):
            s = sub[sub["Features"] == feat].set_index("Model").loc[models]
            ax.bar(x + (i - 0.5) * 0.4, s["Mean"], 0.4, yerr=s["Std"], capsize=4,
                   label=feat, color=colors[feat])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/13_cross_validation.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/13_cross_validation.png")


if __name__ == "__main__":
    main()