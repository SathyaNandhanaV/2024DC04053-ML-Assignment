# ============================================================
# BITS ML Assignment 2 - Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Assignment 2 - Classification Models")

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    models = {}
    model_files = [
        "logistic.pkl",
        "decision_tree.pkl",
        "knn.pkl",
        "naive_bayes.pkl",
        "random_forest.pkl",
        "xgboost.pkl"
    ]

    for file in model_files:
        path = os.path.join("model", file)
        if os.path.exists(path):
            models[file.replace(".pkl", "")] = joblib.load(path)

    return models


models = load_models()

if not models:
    st.error("No models found in model/ folder.")
    st.stop()

# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload test dataset to proceed.")
    st.stop()

data = pd.read_csv(uploaded_file)

st.success("Dataset uploaded successfully!")
st.write("Dataset Shape:", data.shape)

# Assume last column is target
target_column = data.columns[-1]
X_test = data.drop(columns=[target_column])
y_test = data[target_column]

# ============================================================
# MODEL SELECTION
# ============================================================

st.header("Select Model")

model_name = st.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

model = models[model_name]

# ============================================================
# PREDICTION
# ============================================================

st.header("Model Evaluation")

try:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
except:
    st.error("Model failed to predict. Ensure test data matches training features.")
    st.stop()

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
auc = roc_auc_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col1.metric("AUC", f"{auc:.4f}")

col2.metric("Precision", f"{precision:.4f}")
col2.metric("Recall", f"{recall:.4f}")

col3.metric("F1 Score", f"{f1:.4f}")
col3.metric("MCC", f"{mcc:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader("Classification Report")

report = classification_report(y_test, y_pred)
st.text(report)
