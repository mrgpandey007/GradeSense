"""
generate_data.py
-----------------
Generates a synthetic student performance dataset that follows the same
column structure as the real UCI "Student Performance" dataset
(mat2.csv / por2.csv, Cortez & Silva, 2008).

Why synthetic data:
  We don't have the real Kaggle dataset available in this environment yet.
  This script creates 1000 realistic student records with the SAME columns
  and SAME data types as the real dataset, so once the real CSV is dropped
  into data/raw/, no code changes are needed anywhere else in the pipeline.

Output:
  data/raw/student_data.csv
"""

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

N = 1000  # number of student records


def generate_dataset(n=N):
    data = {}

    # --- Demographic / background features ---
    data["school"] = np.random.choice(["GP", "MS"], size=n, p=[0.88, 0.12])
    data["sex"] = np.random.choice(["F", "M"], size=n, p=[0.53, 0.47])
    data["age"] = np.random.randint(15, 20, size=n)
    data["address"] = np.random.choice(["U", "R"], size=n, p=[0.78, 0.22])
    data["famsize"] = np.random.choice(["LE3", "GT3"], size=n, p=[0.29, 0.71])
    data["Pstatus"] = np.random.choice(["T", "A"], size=n, p=[0.90, 0.10])

    # Parents' education (0 = none, 4 = higher education)
    data["Medu"] = np.random.randint(0, 5, size=n)
    data["Fedu"] = np.random.randint(0, 5, size=n)

    jobs = ["teacher", "health", "services", "at_home", "other"]
    data["Mjob"] = np.random.choice(jobs, size=n, p=[0.08, 0.07, 0.24, 0.17, 0.44])
    data["Fjob"] = np.random.choice(jobs, size=n, p=[0.04, 0.04, 0.18, 0.06, 0.68])

    data["reason"] = np.random.choice(
        ["home", "reputation", "course", "other"], size=n, p=[0.24, 0.26, 0.38, 0.12]
    )
    data["guardian"] = np.random.choice(
        ["mother", "father", "other"], size=n, p=[0.68, 0.24, 0.08]
    )

    data["traveltime"] = np.random.choice([1, 2, 3, 4], size=n, p=[0.45, 0.32, 0.15, 0.08])
    data["studytime"] = np.random.choice([1, 2, 3, 4], size=n, p=[0.20, 0.45, 0.22, 0.13])
    data["failures"] = np.random.choice([0, 1, 2, 3], size=n, p=[0.72, 0.15, 0.08, 0.05])

    data["schoolsup"] = np.random.choice(["yes", "no"], size=n, p=[0.11, 0.89])
    data["famsup"] = np.random.choice(["yes", "no"], size=n, p=[0.61, 0.39])
    data["paid"] = np.random.choice(["yes", "no"], size=n, p=[0.42, 0.58])
    data["activities"] = np.random.choice(["yes", "no"], size=n, p=[0.51, 0.49])
    data["nursery"] = np.random.choice(["yes", "no"], size=n, p=[0.80, 0.20])
    data["higher"] = np.random.choice(["yes", "no"], size=n, p=[0.95, 0.05])
    data["internet"] = np.random.choice(["yes", "no"], size=n, p=[0.83, 0.17])
    data["romantic"] = np.random.choice(["yes", "no"], size=n, p=[0.33, 0.67])

    data["famrel"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.03, 0.05, 0.17, 0.42, 0.33])
    data["freetime"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.32, 0.29, 0.19])
    data["goout"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.22, 0.30, 0.22, 0.16])
    data["Dalc"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.60, 0.20, 0.10, 0.06, 0.04])
    data["Walc"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.35, 0.22, 0.18, 0.14, 0.11])
    data["health"] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.10, 0.15, 0.21, 0.44])

    # Absences: right-skewed (most students have few, some have many)
    data["absences"] = np.random.poisson(lam=5, size=n)
    data["absences"] = np.clip(data["absences"], 0, 75)

    df = pd.DataFrame(data)

    # --- Build grades with realistic relationships to the features above ---
    # Base score influenced by studytime, failures, absences, parental education,
    # school support, internet access and free-time/going-out balance.
    base = (
        10
        + (df["studytime"] * 1.3)
        - (df["failures"] * 2.5)
        - (df["absences"] * 0.08)
        + ((df["Medu"] + df["Fedu"]) * 0.35)
        + (df["schoolsup"].eq("yes") * 0.8)
        + (df["internet"].eq("yes") * 0.5)
        + (df["higher"].eq("yes") * 1.0)
        - (df["goout"] * 0.3)
        - (df["Dalc"] * 0.2)
        - (df["Walc"] * 0.15)
        + (df["famrel"] * 0.2)
    )

    noise = np.random.normal(0, 2.0, size=n)
    g1 = np.clip(base + noise, 0, 20).round().astype(int)

    noise2 = np.random.normal(0, 1.5, size=n)
    g2 = np.clip(g1 * 0.9 + base * 0.1 + noise2, 0, 20).round().astype(int)

    noise3 = np.random.normal(0, 1.2, size=n)
    g3 = np.clip(g2 * 0.85 + g1 * 0.1 + base * 0.05 + noise3, 0, 20).round().astype(int)

    df["G1"] = g1
    df["G2"] = g2
    df["G3"] = g3

    return df


if __name__ == "__main__":
    df = generate_dataset(N)
    output_path = "data/raw/student_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records with {len(df.columns)} columns.")
    print(f"Saved to: {output_path}")
    print("\nColumn names:")
    print(list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nG3 (final grade) distribution:")
    print(df["G3"].describe())
