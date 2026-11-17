"""
eda.py
------
Exploratory Data Analysis on the (raw, pre-encoding) student dataset.
We use the RAW data here (not the scaled/encoded processed data) because
plots are more readable with actual category names (e.g. "GP"/"MS")
and actual grade values (0-20) rather than scaled numbers.

Input:
  data/raw/student_data.csv

Output (all saved to outputs/plots/):
  01_grade_distribution.png     - histogram of G3
  02_pass_fail_split.png        - pass vs fail count
  03_correlation_heatmap.png    - correlation of numeric features with G3
  04_studytime_vs_grade.png     - boxplot: studytime vs G3
  05_absences_vs_grade.png      - scatter: absences vs G3
  06_failures_vs_grade.png      - boxplot: past failures vs G3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RAW_PATH = "data/raw/student_data.csv"
PLOTS_DIR = "outputs/plots"
PASS_THRESHOLD = 10

sns.set_style("whitegrid")


def load_data():
    df = pd.read_csv(RAW_PATH)
    df["pass_fail"] = np.where(df["G3"] >= PASS_THRESHOLD, "Pass", "Fail")
    return df


def plot_grade_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["G3"], bins=20, kde=True, color="steelblue")
    plt.axvline(PASS_THRESHOLD, color="red", linestyle="--", label=f"Pass threshold ({PASS_THRESHOLD})")
    plt.title("Distribution of Final Grades (G3)")
    plt.xlabel("Final Grade (G3)")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_grade_distribution.png", dpi=150)
    plt.close()
    print("Saved: 01_grade_distribution.png")


def plot_pass_fail_split(df):
    plt.figure(figsize=(6, 5))
    counts = df["pass_fail"].value_counts()
    colors = ["#4CAF50" if idx == "Pass" else "#E53935" for idx in counts.index]
    plt.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.title("Pass vs Fail Split")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_pass_fail_split.png", dpi=150)
    plt.close()
    print("Saved: 02_pass_fail_split.png")


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: 03_correlation_heatmap.png")

    # Also print top correlations with G3 specifically — useful for the report
    g3_corr = corr["G3"].drop(["G3", "G1", "G2"]).sort_values(ascending=False)
    print("\nTop features correlated with final grade (G3):")
    print(g3_corr.head(10))
    print("\nMost negatively correlated with G3:")
    print(g3_corr.tail(5))


def plot_studytime_vs_grade(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="studytime", y="G3", hue="studytime", palette="Blues", legend=False)
    plt.title("Study Time vs Final Grade")
    plt.xlabel("Weekly Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_studytime_vs_grade.png", dpi=150)
    plt.close()
    print("Saved: 04_studytime_vs_grade.png")


def plot_absences_vs_grade(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="absences", y="G3", hue="pass_fail",
                     palette={"Pass": "#4CAF50", "Fail": "#E53935"}, alpha=0.6)
    plt.title("Absences vs Final Grade")
    plt.xlabel("Number of Absences")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_absences_vs_grade.png", dpi=150)
    plt.close()
    print("Saved: 05_absences_vs_grade.png")


def plot_failures_vs_grade(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="failures", y="G3", hue="failures", palette="Reds", legend=False)
    plt.title("Past Class Failures vs Final Grade")
    plt.xlabel("Number of Past Failures")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_failures_vs_grade.png", dpi=150)
    plt.close()
    print("Saved: 06_failures_vs_grade.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_data()

    print(f"Dataset shape: {df.shape}\n")
    print("Summary statistics for key numeric columns:")
    print(df[["studytime", "absences", "failures", "G1", "G2", "G3"]].describe())
    print()

    plot_grade_distribution(df)
    plot_pass_fail_split(df)
    plot_correlation_heatmap(df)
    plot_studytime_vs_grade(df)
    plot_absences_vs_grade(df)
    plot_failures_vs_grade(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
