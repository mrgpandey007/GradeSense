"""
evaluate.py
-----------
Loads the trained models and their held-out test sets, then reports:
  Regression (predicting G3):
    - R2 score, RMSE, MAE  (Linear Regression vs Random Forest Regressor)
    - Actual vs Predicted scatter plot
  Classification (predicting pass_fail):
    - Accuracy, Precision, Recall, F1
    - Confusion matrix plots (Logistic Regression vs Random Forest Classifier)
  Feature importance:
    - Which features the Random Forest models rely on most

Input:
  models/*.joblib  (from train_models.py)

Output:
  outputs/plots/07_actual_vs_predicted.png
  outputs/plots/08_confusion_matrices.png
  outputs/plots/09_feature_importance_regression.png
  outputs/plots/10_feature_importance_classification.png
  outputs/reports/metrics_summary.txt
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

MODELS_DIR = "models"
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"

sns.set_style("whitegrid")


def evaluate_regression():
    lr = joblib.load(f"{MODELS_DIR}/linear_regression.joblib")
    rf = joblib.load(f"{MODELS_DIR}/random_forest_regressor.joblib")
    X_test, y_test = joblib.load(f"{MODELS_DIR}/regression_test_set.joblib")

    results = {}
    preds = {}
    for name, model in [("Linear Regression", lr), ("Random Forest Regressor", rf)]:
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"\n[{name}]")
        print(f"  R2 Score : {r2:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        print(f"  MAE      : {mae:.4f}")

    # Actual vs Predicted plot (both models side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
        lims = [0, 20]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual G3")
        ax.set_ylabel("Predicted G3")
        ax.set_title(f"{name}\nR2={results[name]['R2']:.3f}, RMSE={results[name]['RMSE']:.2f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_actual_vs_predicted.png", dpi=150)
    plt.close()
    print("\nSaved: 07_actual_vs_predicted.png")

    # Feature importance (Random Forest Regressor only — Linear Regression
    # coefficients aren't directly comparable without more normalization work)
    importances = pd.Series(rf.feature_importances_, index=X_test.columns)
    top_features = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index,
                palette="Blues_r", legend=False)
    plt.title("Top 10 Feature Importances — Random Forest Regressor (predicting G3)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/09_feature_importance_regression.png", dpi=150)
    plt.close()
    print("Saved: 09_feature_importance_regression.png")

    return results, top_features


def evaluate_classification():
    logreg = joblib.load(f"{MODELS_DIR}/logistic_regression.joblib")
    rfc = joblib.load(f"{MODELS_DIR}/random_forest_classifier.joblib")
    X_test, y_test = joblib.load(f"{MODELS_DIR}/classification_test_set.joblib")

    results = {}
    cms = {}
    for name, model in [("Logistic Regression", logreg), ("Random Forest Classifier", rfc)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        cms[name] = cm

        print(f"\n[{name}]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
        ax.set_title(f"{name}\nAccuracy={results[name]['Accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_confusion_matrices.png", dpi=150)
    plt.close()
    print("\nSaved: 08_confusion_matrices.png")

    # Feature importance (Random Forest Classifier)
    importances = pd.Series(rfc.feature_importances_, index=X_test.columns)
    top_features = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index,
                palette="Greens_r", legend=False)
    plt.title("Top 10 Feature Importances — Random Forest Classifier (predicting pass/fail)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/10_feature_importance_classification.png", dpi=150)
    plt.close()
    print("Saved: 10_feature_importance_classification.png")

    return results, top_features


def save_metrics_report(reg_results, clf_results, reg_top_features, clf_top_features):
    lines = []
    lines.append("=" * 60)
    lines.append("MODEL EVALUATION SUMMARY")
    lines.append("=" * 60)

    lines.append("\n--- REGRESSION (predicting G3, final grade 0-20) ---")
    for name, m in reg_results.items():
        lines.append(f"\n{name}:")
        lines.append(f"  R2 Score : {m['R2']:.4f}")
        lines.append(f"  RMSE     : {m['RMSE']:.4f}")
        lines.append(f"  MAE      : {m['MAE']:.4f}")

    lines.append("\n\n--- CLASSIFICATION (predicting pass_fail) ---")
    for name, m in clf_results.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Accuracy  : {m['Accuracy']:.4f}")
        lines.append(f"  Precision : {m['Precision']:.4f}")
        lines.append(f"  Recall    : {m['Recall']:.4f}")
        lines.append(f"  F1 Score  : {m['F1']:.4f}")

    lines.append("\n\n--- TOP FEATURES (Random Forest Regressor, for G3) ---")
    for feat, val in reg_top_features.items():
        lines.append(f"  {feat}: {val:.4f}")

    lines.append("\n\n--- TOP FEATURES (Random Forest Classifier, for pass_fail) ---")
    for feat, val in clf_top_features.items():
        lines.append(f"  {feat}: {val:.4f}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = f"{REPORTS_DIR}/metrics_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMetrics summary saved to: {report_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Evaluating regression models (predicting G3)...")
    reg_results, reg_top_features = evaluate_regression()

    print("\n" + "=" * 60)
    print("Evaluating classification models (predicting pass_fail)...")
    clf_results, clf_top_features = evaluate_classification()

    save_metrics_report(reg_results, clf_results, reg_top_features, clf_top_features)


if __name__ == "__main__":
    main()
