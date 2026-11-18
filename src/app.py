"""
app.py  (v2 - interactive)
---------------------------
Streamlit web app for the Student Performance Prediction System.

Features:
  - Sidebar: choose the mode (with / without previous grades G1, G2) and one of
    5 models; shows that model's held-out test metrics.
  - Live prediction: results update instantly as inputs change (no submit button).
  - One-click sample profiles (Strong / Average / At-risk student).
  - Pass probability bar + risk level, and predictions from ALL models side by side.
  - Compare Models tab: metrics table + chart for the selected mode.

Run with:
  streamlit run src/app.py

Needs (created by the earlier scripts):
  models/model_bundle.joblib          (train_all_models.py)
  data/processed/label_encoders.joblib, data/processed/scaler.joblib  (preprocess.py)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = BASE_DIR / "models" / "model_bundle.joblib"
ENCODERS_PATH = BASE_DIR / "data" / "processed" / "label_encoders.joblib"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.joblib"

MULTI_COLS = ["Mjob", "Fjob", "reason", "guardian"]
PCT = 5  # models use the 0-20 scale; the app shows percentages (x5)
JOBS = ["teacher", "health", "services", "at_home", "other"]

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# Default values + sample profiles (all widgets are keyed, so profiles can set them)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
    "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes",
    "paid": "no", "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1,
    "health": 4, "absences": 5, "G1": 60, "G2": 60,
}

PROFILES = {
    "Strong student": {
        "studytime": 4, "failures": 0, "absences": 1, "Medu": 4, "Fedu": 4, "higher": "yes",
        "internet": "yes", "goout": 2, "Dalc": 1, "Walc": 1, "G1": 85, "G2": 90, "paid": "yes",
    },
    "Average student": {
        "studytime": 2, "failures": 0, "absences": 5, "Medu": 2, "Fedu": 2, "higher": "yes",
        "internet": "yes", "goout": 3, "Dalc": 1, "Walc": 2, "G1": 60, "G2": 60, "paid": "no",
    },
    "At-risk student": {
        "studytime": 1, "failures": 2, "absences": 18, "Medu": 1, "Fedu": 1, "higher": "no",
        "internet": "no", "goout": 5, "Dalc": 3, "Walc": 4, "G1": 35, "G2": 30, "paid": "no",
    },
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def apply_profile(name):
    for key, value in {**DEFAULTS, **PROFILES[name]}.items():
        st.session_state[key] = value


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    bundle = joblib.load(BUNDLE_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    return bundle, encoders, scaler


def build_features(raw, mode_key, bundle, encoders, scaler):
    """Encode + scale one student's raw inputs exactly like training, then select
    the columns used by the chosen mode (with / without G1, G2)."""
    full_cols = bundle["with"]["features"]  # superset: includes G1, G2
    df = pd.DataFrame([raw])

    for col, le in encoders.items():          # binary / two-category columns
        if col in df.columns:
            df[col] = le.transform(df[col])

    for col in MULTI_COLS:                    # manual one-hot (single row safe)
        chosen = df[col].iloc[0]
        for fc in full_cols:
            if fc.startswith(f"{col}_"):
                df[fc] = 1 if fc == f"{col}_{chosen}" else 0
        df = df.drop(columns=[col])

    for col in full_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[full_cols].astype(float)

    scale_cols = list(scaler.feature_names_in_)
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df[bundle[mode_key]["features"]]


def risk_label(pass_prob):
    if pass_prob >= 0.80:
        return "Low risk", "success"
    if pass_prob >= 0.55:
        return "Medium risk", "warning"
    return "High risk", "error"


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    try:
        bundle, encoders, scaler = load_artifacts()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.info("Run preprocess.py and train_all_models.py first, then restart the app.")
        st.stop()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.title("⚙️ Settings")
        mode_label = st.radio(
            "Prediction mode",
            ["With previous grades (G1, G2)", "Early warning (without grades)"],
            help="Early-warning mode works at the start of a term, before any grades exist, "
                 "but is less accurate.",
        )
        mode_key = "with" if mode_label.startswith("With") else "without"

        reg_name = st.selectbox("Regression model (final grade)", list(bundle[mode_key]["regression"]), index=1)
        clf_name = st.selectbox("Classification model (pass / fail)", list(bundle[mode_key]["classification"]), index=1)

        st.markdown("---")
        st.subheader("Model performance")
        st.caption("Measured on a held-out test set.")
        rm = bundle[mode_key]["regression"][reg_name]["metrics"]
        cm = bundle[mode_key]["classification"][clf_name]["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("R²", f"{rm['R2']:.2f}")
        c2.metric("RMSE (%)", f"{rm['RMSE'] * PCT:.1f}")
        c3, c4 = st.columns(2)
        c3.metric("Accuracy", f"{cm['Accuracy']*100:.0f}%")
        c4.metric("Fail recall", f"{cm['Fail Recall']*100:.0f}%")

    st.title("🎓 Student Performance Predictor")
    st.caption("Predict a student's final grade (as a percentage, 0-100%) and pass/fail status. Results update live.")

    tab_predict, tab_compare, tab_about = st.tabs(["🔮 Predict", "📊 Compare Models", "ℹ️ About"])

    # ====================== PREDICT TAB ======================
    with tab_predict:
        st.write("**Quick start:** load a sample profile, then tweak any value.")
        b1, b2, b3, _ = st.columns([1, 1, 1, 3])
        b1.button("💪 Strong student", on_click=apply_profile, args=("Strong student",), width="stretch")
        b2.button("🙂 Average student", on_click=apply_profile, args=("Average student",), width="stretch")
        b3.button("⚠️ At-risk student", on_click=apply_profile, args=("At-risk student",), width="stretch")

        left, right = st.columns([3, 2], gap="large")

        with left:
            with st.expander("👨‍👩‍👧 Background", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.selectbox("School", ["GP", "MS"], key="school")
                    st.selectbox("Sex", ["F", "M"], key="sex")
                    st.slider("Age", 15, 22, key="age")
                    st.selectbox("Address type", ["U", "R"], key="address", help="U = Urban, R = Rural")
                    st.selectbox("Family size", ["LE3", "GT3"], key="famsize", help="LE3 = up to 3, GT3 = more than 3")
                    st.selectbox("Parents' status", ["T", "A"], key="Pstatus", help="T = together, A = apart")
                with c2:
                    st.slider("Mother's education (0-4)", 0, 4, key="Medu")
                    st.slider("Father's education (0-4)", 0, 4, key="Fedu")
                    st.selectbox("Mother's job", JOBS, key="Mjob")
                    st.selectbox("Father's job", JOBS, key="Fjob")
                    st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"], key="reason")
                    st.selectbox("Guardian", ["mother", "father", "other"], key="guardian")

            with st.expander("📚 Study & school life", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.slider("Travel time (1 = <15 min ... 4 = >1 hr)", 1, 4, key="traveltime")
                    st.slider("Weekly study time (1 = <2 hr ... 4 = >10 hr)", 1, 4, key="studytime")
                    st.slider("Past class failures", 0, 3, key="failures")
                    st.slider("Absences", 0, 75, key="absences")
                    st.selectbox("Extra paid classes", ["yes", "no"], key="paid")
                    st.selectbox("Extra educational support", ["yes", "no"], key="schoolsup")
                with c2:
                    st.selectbox("Family educational support", ["yes", "no"], key="famsup")
                    st.selectbox("Extra-curricular activities", ["yes", "no"], key="activities")
                    st.selectbox("Attended nursery school", ["yes", "no"], key="nursery")
                    st.selectbox("Wants higher education", ["yes", "no"], key="higher")
                    st.selectbox("Internet at home", ["yes", "no"], key="internet")
                    st.selectbox("In a romantic relationship", ["yes", "no"], key="romantic")

            with st.expander("🌿 Lifestyle", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.slider("Family relationship quality (1-5)", 1, 5, key="famrel")
                    st.slider("Free time after school (1-5)", 1, 5, key="freetime")
                    st.slider("Going out with friends (1-5)", 1, 5, key="goout")
                with c2:
                    st.slider("Workday alcohol (1-5)", 1, 5, key="Dalc")
                    st.slider("Weekend alcohol (1-5)", 1, 5, key="Walc")
                    st.slider("Health status (1-5)", 1, 5, key="health")

            if mode_key == "with":
                with st.expander("📝 Previous grades", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.slider("First term grade (G1) %", 0, 100, key="G1")
                    c2.slider("Second term grade (G2) %", 0, 100, key="G2")
            else:
                st.info("Early-warning mode: previous grades (G1, G2) are not used, "
                        "so this works from day one of the term.")

        # ---------------- Live results ----------------
        raw = {k: st.session_state[k] for k in DEFAULTS}
        raw["G1"] = raw["G1"] / PCT   # percent -> 0-20 scale used in training
        raw["G2"] = raw["G2"] / PCT
        X = build_features(raw, mode_key, bundle, encoders, scaler)

        reg_model = bundle[mode_key]["regression"][reg_name]["model"]
        clf_model = bundle[mode_key]["classification"][clf_name]["model"]
        grade = float(np.clip(reg_model.predict(X)[0], 0, 20)) * PCT   # shown as %
        pass_prob = float(clf_model.predict_proba(X)[0][list(clf_model.classes_).index(1)])
        is_pass = pass_prob >= 0.5
        risk_text, risk_kind = risk_label(pass_prob)

        with right:
            st.subheader("Prediction")
            m1, m2 = st.columns(2)
            m1.metric("Final grade", f"{grade:.1f}%")
            m2.metric("Status", "✅ PASS" if is_pass else "❌ FAIL")

            st.caption("Pass mark: 50%")
            st.write(f"**Pass probability: {pass_prob*100:.0f}%**")
            st.progress(pass_prob)
            getattr(st, risk_kind)(f"{risk_text}")

            if not is_pass or risk_kind != "success":
                st.caption("Tip: try raising study time, lowering absences, or adding extra "
                           "support and watch the prediction change.")

            st.markdown("---")
            st.subheader("All models, same student")
            rows = []
            for name in bundle[mode_key]["regression"]:
                g = float(np.clip(bundle[mode_key]["regression"][name]["model"].predict(X)[0], 0, 20)) * PCT
                rows.append({"Model": name, "Predicted grade (%)": round(g, 1)})
            grade_df = pd.DataFrame(rows).set_index("Model")
            st.bar_chart(grade_df, height=220)

            rows = []
            for name in bundle[mode_key]["classification"]:
                mdl = bundle[mode_key]["classification"][name]["model"]
                p = float(mdl.predict_proba(X)[0][list(mdl.classes_).index(1)])
                rows.append({"Model": name, "Pass probability (%)": round(p * 100, 0)})
            st.bar_chart(pd.DataFrame(rows).set_index("Model"), height=220)

    # ====================== COMPARE TAB ======================
    with tab_compare:
        st.subheader(f"Model comparison ({'with' if mode_key == 'with' else 'without'} G1/G2)")
        st.caption("All numbers are measured on a held-out test set.")

        reg_rows = [{"Model": n, "R²": round(v["metrics"]["R2"], 3), "RMSE (%)": round(v["metrics"]["RMSE"] * PCT, 1)}
                    for n, v in bundle[mode_key]["regression"].items()]
        clf_rows = [{"Model": n, "Accuracy": round(v["metrics"]["Accuracy"], 3),
                     "F1": round(v["metrics"]["F1"], 3), "Fail recall": round(v["metrics"]["Fail Recall"], 3)}
                    for n, v in bundle[mode_key]["classification"].items()]
        reg_df, clf_df = pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Regression (predicting final grade)**")
            st.dataframe(reg_df, hide_index=True, width="stretch")
            st.bar_chart(reg_df.set_index("Model")[["R²"]], height=260)
        with c2:
            st.write("**Classification (predicting pass / fail)**")
            st.dataframe(clf_df, hide_index=True, width="stretch")
            st.bar_chart(clf_df.set_index("Model")[["Accuracy"]], height=260)

        st.markdown("---")
        st.write("**With vs without previous grades**")
        rows = []
        for key, label in [("with", "With G1/G2"), ("without", "Without G1/G2")]:
            best_r = max(bundle[key]["regression"].items(), key=lambda kv: kv[1]["metrics"]["R2"])
            best_c = max(bundle[key]["classification"].items(), key=lambda kv: kv[1]["metrics"]["Accuracy"])
            rows.append({
                "Setting": label,
                "Best regression model": best_r[0],
                "Best R²": round(best_r[1]["metrics"]["R2"], 3),
                "Best classification model": best_c[0],
                "Best accuracy": round(best_c[1]["metrics"]["Accuracy"], 3),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Removing G1/G2 lowers accuracy because previous grades are the strongest predictors, "
                   "but the early-warning mode can flag at-risk students before any grades exist.")

    # ====================== ABOUT TAB ======================
    with tab_about:
        st.subheader("About this project")
        st.markdown(
            """
**AI-Based Student Performance Prediction System** — B.Tech CSE (AI/ML) project,
Dev Bhoomi Uttarakhand University.

- **Data:** UCI Student Performance dataset (Cortez & Silva, 2008), 395 students, 33 features.
- **Tasks:** regression (final grade, shown as a percentage) and classification (pass/fail, pass mark 50%).
- **Models:** Linear / Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN.
- **Modes:** with previous grades (more accurate) or early warning without grades.
- **Note:** predictions are a decision-support aid for teachers, not a final judgement
  about any student.
            """
        )


if __name__ == "__main__":
    main()