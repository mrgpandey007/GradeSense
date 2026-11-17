"""
preprocess.py
--------------
Loads the raw student dataset (data/raw/student_data.csv), cleans it,
encodes categorical columns, derives the pass_fail target, and scales
numeric features. Saves the processed dataset for model training.

Input:
  data/raw/student_data.csv

Output:
  data/processed/student_data_processed.csv   (encoded + scaled, ready for ML)
  data/processed/label_encoders.joblib        (saved encoders, for decoding/reuse)
  data/processed/scaler.joblib                (saved scaler, needed at prediction time)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_PATH = "data/raw/student_data.csv"
PROCESSED_DIR = "data/processed"
PASS_THRESHOLD = 10  # G3 >= 10 => pass (UCI convention, grades are 0-20)


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def check_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("No missing values found.")
    else:
        print("Missing values found:")
        print(missing)
        # Simple strategy: numeric -> median, categorical -> mode
        for col in missing.index:
            if df[col].dtype in [np.int64, np.float64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        print("Missing values filled (numeric->median, categorical->mode).")
    return df


def add_target_columns(df):
    # Classification target: pass/fail based on final grade G3
    df["pass_fail"] = (df["G3"] >= PASS_THRESHOLD).astype(int)  # 1 = pass, 0 = fail
    print(f"pass_fail distribution:\n{df['pass_fail'].value_counts()}")
    return df


def encode_categoricals(df):
    # Binary yes/no and two-category columns -> Label Encoding (0/1)
    binary_cols = [
        "school", "sex", "address", "famsize", "Pstatus",
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic",
    ]

    # Multi-category columns -> One-Hot Encoding
    multi_cols = ["Mjob", "Fjob", "reason", "guardian"]

    encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    # Convert one-hot bool columns to int (0/1) for consistency
    dummy_cols = [c for c in df.columns if c.split("_")[0] in
                  [m for m in multi_cols]]
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    return df, encoders


def scale_numeric(df):
    # Scale numeric features (not the targets G1/G2/G3/pass_fail — keep those interpretable)
    exclude = ["G1", "G2", "G3", "pass_fail"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [c for c in numeric_cols if c not in exclude]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, scaler


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data()
    df = check_missing(df)
    df = add_target_columns(df)
    df, encoders = encode_categoricals(df)
    df, scaler = scale_numeric(df)

    output_path = os.path.join(PROCESSED_DIR, "student_data_processed.csv")
    df.to_csv(output_path, index=False)

    joblib.dump(encoders, os.path.join(PROCESSED_DIR, "label_encoders.joblib"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))

    print(f"\nProcessed dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")


if __name__ == "__main__":
    main()
