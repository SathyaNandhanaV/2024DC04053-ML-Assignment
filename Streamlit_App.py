import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score

from model.models import get_model

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="2024DC04053-ML Assignment-2",
    layout="wide",
    page_icon="📊"
)

# ----------------------------
# Custom Styling
# ----------------------------

st.markdown("""
<style>
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------

st.title("📊 Bank Marketing ML Dashboard")
st.markdown("### Clean Model Evaluation & Comparison")

# ----------------------------
# Sidebar Controls
# ----------------------------

st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")

model_name = st.sidebar.selectbox(
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

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# ----------------------------
# Main App
# ----------------------------

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if target_column:

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        with st.spinner("Training model..."):
            model = get_model(model_name, X_train)
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # ----------------------------
        # Metrics Calculation
        # ----------------------------

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = np.mean(cv_scores)

        # ----------------------------
        # Metric Cards
        # ----------------------------

        st.subheader("📈 Model Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("CV Accuracy", f"{cv_mean:.4f}")

        # ----------------------------
        # Confusion Matrix
        # ----------------------------

        st.subheader("🧮 Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ----------------------------
        # Classification Report
        # ----------------------------

        st.subheader("📋 Detailed Classification Report")

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df, use_container_width=True)
