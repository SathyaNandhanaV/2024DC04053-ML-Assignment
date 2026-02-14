import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Adult Income ML App", layout="wide")
st.title("💼 Adult Income Classification Dashboard")
model_option = st.selectbox("Select Model",["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"])
uploaded_file = st.file_uploader("Upload Test CSV File", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df.replace("?", None, inplace=True)
    if "income" not in df.columns:
        st.error("Target column 'income' not found.")
    else:
        X = df.drop("income", axis=1)
        y = df["income"].map({"<=50K": 0, ">50K": 1})
        model = joblib.load(f"models/{model_option}.pkl")
        preds = model.predict(X)
        st.subheader("📊 Classification Report")
        st.text(classification_report(y, preds))
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y, preds)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)
        st.success("Prediction Completed Successfully ✅")
