# AI-Based Student Performance Prediction System

A machine learning project that predicts a student's **final grade (G3, 0-20)** and **pass/fail status** from academic, behavioral, and family-background factors such as study time, past failures, absences, parental education, and previous term grades.

Built as a B.Tech CSE (AI/ML) college project at Dev Bhoomi Uttarakhand University.

## Problem Statement

Student performance is usually assessed only after final exams, which is often too late to help students who are struggling. This project uses machine learning to flag at-risk students early so that teachers can intervene in time.

## Dataset

- **Source:** UCI Student Performance dataset (Cortez & Silva, 2008), downloaded from [Kaggle](https://www.kaggle.com/datasets/henryshan/student-performance-prediction)
- **File used:** `mat2.csv` (Mathematics course), saved as `data/raw/student_data.csv`
- **Size:** 395 student records, 33 features
- **Targets:**
  - `G3`: final grade (regression)
  - `pass_fail`: derived as `G3 >= 10` (classification)

## Models Used

| Task | Models |
|---|---|
| Regression (predict G3) | Linear Regression, Random Forest Regressor |
| Classification (predict pass/fail) | Logistic Regression, Random Forest Classifier |

Evaluation metrics: R², RMSE, MAE (regression); Accuracy, Precision, Recall, F1-score, Confusion Matrix (classification). Feature importance is computed from the Random Forest models.

## Project Structure

```
Project/
├── data/
│   ├── raw/                 # original dataset (student_data.csv)
│   └── processed/           # encoded + scaled data, saved encoders/scaler
├── src/
│   ├── generate_data.py     # synthetic data generator (for testing only)
│   ├── preprocess.py        # cleaning, encoding, scaling, target creation
│   ├── eda.py               # exploratory data analysis plots
│   ├── train_models.py      # trains and saves the 4 models
│   ├── evaluate.py          # metrics, confusion matrix, feature importance
│   └── app.py               # Streamlit web demo
├── models/                  # trained models (.joblib)
├── outputs/
│   ├── plots/               # EDA and evaluation graphs
│   └── reports/             # metrics_summary.txt
├── .gitignore
└── README.md
```

## Setup and Run

**1. Install dependencies**

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

**2. Run the pipeline in order** (from the project root folder)

```
python src/preprocess.py
python src/eda.py
python src/train_models.py
python src/evaluate.py
```

**3. Launch the web demo**

```
streamlit run src/app.py
```

Fill in the student details in the form and click **Predict Performance** to get the predicted final grade and pass/fail status with a confidence score.

## Results

Detailed metrics are saved to `outputs/reports/metrics_summary.txt` after running `evaluate.py`, and all plots are saved in `outputs/plots/`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit

## Team

- Gautam Pandey (2304050024)
- Rohit Biswas (2304051008)
- Adarsh Raj (2304050006)

**Supervisor:** Ms. Vijayshree Khugshal

## Reference

P. Cortez and A. Silva, "Using Data Mining to Predict Secondary School Student Performance," in *Proc. 5th Future Business Technology Conference (FUBUTEC)*, Porto, 2008, pp. 5-12.