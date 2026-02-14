import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from model.logistic_regression import get_model as lr_model
from model.decision_tree import get_model as dt_model
from model.knn import get_model as knn_model
from model.naive_bayes import get_model as nb_model
from model.random_forest import get_model as rf_model

st.set_page_config(page_title="Bank ML Classifier", layout="wide")

st.title("📊 Bank Marketing Classification App")

MODEL_MAP = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
}

model_option = st.selectbox("Select Model", list(MODEL_MAP.keys()))

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.write("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        with st.spinner("Training model..."):
            model = MODEL_MAP[model_option](X)
            model.fit(X, y)

        preds = model.predict(X)

        st.subheader("📈 Classification Report")
        st.text(classification_report(y, preds))

        st.subheader("🧮 Confusion Matrix")
        st.write(confusion_matrix(y, preds))
