import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from model.models import get_model


# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Bank ML Dashboard", layout="wide")
st.title("📊 Bank Marketing ML Dashboard")
st.markdown("Configure model settings and click **Train Model** to run.")

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Speed optimization
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    # ---------------------------
    # FORM SECTION
    # ---------------------------
    with st.form("model_form"):

        st.markdown("### ⚙️ Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            target_column = st.selectbox("Select Target Column", df.columns)

        with col2:
            model_name = st.selectbox(
                "Select Model",
                [
                    "Logistic Regression",
                    "Decision Tree",
                    "KNN",
                    "Naive Bayes",
                    "Random Forest",
                    "XGBoost"
                ]
            )

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        submit_button = st.form_submit_button("🚀 Train Model")

    # ---------------------------
    # TRAIN ONLY AFTER BUTTON
    # ---------------------------
    if submit_button:

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )

        with st.spinner("Training model... Please wait..."):

            model = get_model(model_name, X_train)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
            else:
                auc = None

            cv_scores = cross_val_score(
                get_model(model_name, X_train),
                X,
                y,
                cv=3,
                n_jobs=-1
            )

        # ---------------------------
        # METRICS
        # ---------------------------
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        st.markdown("## 📈 Model Performance")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("MCC", f"{mcc:.4f}")
        col6.metric("CV Accuracy", f"{cv_scores.mean():.4f}")

        if auc is not None:
            st.metric("ROC AUC", f"{auc:.4f}")

        # ---------------------------
        # CONFUSION MATRIX
        # ---------------------------
        st.markdown("## 🧮 Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=1,
            linecolor="gray"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ---------------------------
        # CLASSIFICATION REPORT
        # ---------------------------
        st.markdown("## 📋 Classification Report")

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4), width="stretch")
