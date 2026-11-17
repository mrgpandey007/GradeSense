"""
app.py
------
Streamlit web app: takes student details as input via a form and predicts
  1. Final Grade (G3, 0-20)      -> using Random Forest Regressor
  2. Pass / Fail                  -> using Random Forest Classifier

Run with:
  streamlit run src/app.py

Requires (already trained by earlier scripts):
  models/random_forest_regressor.joblib
  models/random_forest_classifier.joblib
  models/feature_columns.joblib
  data/processed/label_encoders.joblib
  data/processed/scaler.joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_artifacts():
    reg_model = joblib.load(f"{MODELS_DIR}/random_forest_regressor.joblib")
    clf_model = joblib.load(f"{MODELS_DIR}/random_forest_classifier.joblib")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.joblib")
    encoders = joblib.load(f"{PROCESSED_DIR}/label_encoders.joblib")
    scaler = joblib.load(f"{PROCESSED_DIR}/scaler.joblib")
    return reg_model, clf_model, feature_columns, encoders, scaler


def build_input_row(raw_input, feature_columns, encoders, scaler):
    """
    Takes a dict of raw form inputs (human-readable values), applies the
    SAME encoding + scaling used during training, and returns a single-row
    DataFrame with columns matching feature_columns exactly.
    """
    df = pd.DataFrame([raw_input])

    # Apply label encoders (binary yes/no + two-category columns)
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Manually one-hot encode the multi-category columns by matching against
    # the known training-time column names (feature_columns). We do NOT use
    # pd.get_dummies() here: with only a single row, get_dummies() sees just
    # one category present and (with drop_first behavior baked into the
    # training columns) can silently produce the WRONG dummy columns or none
    # at all. Building columns explicitly from feature_columns avoids that.
    multi_cols = ["Mjob", "Fjob", "reason", "guardian"]
    for col in multi_cols:
        if col in df.columns:
            selected_value = df[col].iloc[0]
            expected_dummy_col = f"{col}_{selected_value}"
            for fc in feature_columns:
                if fc.startswith(f"{col}_"):
                    df[fc] = 1 if fc == expected_dummy_col else 0
            df = df.drop(columns=[col])

    # Add any remaining missing columns as 0, then enforce exact column
    # order to match what the models were trained on
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]  # exact order + set

    # Scale numeric columns the same way training did (exclude G1/G2 targets
    # only if they were excluded during training scaling — here G1, G2 ARE
    # features so they were scaled too; scaler was fit on all non-target
    # numeric cols including G1/G2)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [c for c in numeric_cols if c in scaler.feature_names_in_]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df


def main():
    st.title("🎓 Student Performance Predictor")
    st.write(
        "Fill in student details below to predict their **final grade (G3)** "
        "and **pass/fail** status."
    )

    reg_model, clf_model, feature_columns, encoders, scaler = load_artifacts()

    with st.form("student_form"):
        st.subheader("Background")
        col1, col2 = st.columns(2)
        with col1:
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Sex", ["F", "M"])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address Type", ["U", "R"], help="U=Urban, R=Rural")
            famsize = st.selectbox("Family Size", ["LE3", "GT3"], help="LE3=<=3, GT3=>3")
            Pstatus = st.selectbox("Parent's Cohabitation Status", ["T", "A"], help="T=Together, A=Apart")
        with col2:
            Medu = st.slider("Mother's Education (0=none, 4=higher)", 0, 4, 2)
            Fedu = st.slider("Father's Education (0=none, 4=higher)", 0, 4, 2)
            Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
            Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
            reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])

        st.subheader("Study & School Life")
        col3, col4 = st.columns(2)
        with col3:
            traveltime = st.slider("Travel Time (1=<15min ... 4=>1hr)", 1, 4, 1)
            studytime = st.slider("Weekly Study Time (1=<2hr ... 4=>10hr)", 1, 4, 2)
            failures = st.slider("Past Class Failures", 0, 3, 0)
            schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
            famsup = st.selectbox("Family Educational Support", ["yes", "no"])
            paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
        with col4:
            activities = st.selectbox("Extra-curricular Activities", ["yes", "no"])
            nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
            higher = st.selectbox("Wants Higher Education", ["yes", "no"])
            internet = st.selectbox("Internet Access at Home", ["yes", "no"])
            romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])
            absences = st.slider("Number of Absences", 0, 75, 5)

        st.subheader("Lifestyle")
        col5, col6 = st.columns(2)
        with col5:
            famrel = st.slider("Quality of Family Relationships (1-5)", 1, 5, 4)
            freetime = st.slider("Free Time After School (1-5)", 1, 5, 3)
            goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
        with col6:
            Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
            health = st.slider("Current Health Status (1-5)", 1, 5, 4)

        st.subheader("Previous Grades")
        col7, col8 = st.columns(2)
        with col7:
            G1 = st.slider("First Term Grade (G1)", 0, 20, 12)
        with col8:
            G2 = st.slider("Second Term Grade (G2)", 0, 20, 12)

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        raw_input = {
            "school": school, "sex": sex, "age": age, "address": address,
            "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
            "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
            "traveltime": traveltime, "studytime": studytime, "failures": failures,
            "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher,
            "internet": internet, "romantic": romantic, "famrel": famrel,
            "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
            "health": health, "absences": absences, "G1": G1, "G2": G2,
        }

        try:
            X_input = build_input_row(raw_input, feature_columns, encoders, scaler)

            predicted_grade = reg_model.predict(X_input)[0]
            predicted_grade = np.clip(predicted_grade, 0, 20)

            pass_pred = clf_model.predict(X_input)[0]
            pass_proba = clf_model.predict_proba(X_input)[0][1]  # probability of "pass"

            st.markdown("---")
            st.subheader("Prediction Results")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Final Grade (G3)", f"{predicted_grade:.1f} / 20")
            with col_b:
                status = "✅ PASS" if pass_pred == 1 else "❌ FAIL"
                st.metric("Predicted Status", status, f"{pass_proba*100:.1f}% pass probability")

            if pass_pred == 1:
                st.success("This student is predicted to PASS.")
            else:
                st.error("This student is predicted to FAIL. Consider extra support.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "This usually means the trained models/encoders don't match this app's "
                "expected features. Make sure you've run generate_data.py -> preprocess.py "
                "-> train_models.py in order before launching this app."
            )


if __name__ == "__main__":
    main()
