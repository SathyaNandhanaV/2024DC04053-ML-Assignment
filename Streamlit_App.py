import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="ML Assignment 2")

st.title("Machine Learning Assignment 2")

# ============================
# LOAD MODELS (WITHOUT XGBOOST)
# ============================

@st.cache_resource
def load_models():
    models = {}
    files = [
        "logistic.pkl",
        "decision_tree.pkl",
        "knn.pkl",
        "naive_bayes.pkl",
        "random_forest.pkl"
    ]

    for f in files:
        path = os.path.join("model", f)
        if os.path.exists(path):
            models[f.replace(".pkl", "")] = joblib.load(path)

    return models

models = load_models()

if not models:
    st.error("No models found.")
    st.stop()

# ============================
# UPLOAD DATA
# ============================

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

data = pd.read_csv(uploaded_file)

target = data.columns[-1]
X_test = data.drop(columns=[target])
y_test = data[target]

# ============================
# MODEL SELECTION
# ============================

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ============================
# EVALUATION
# ============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy:.4f}")

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))
