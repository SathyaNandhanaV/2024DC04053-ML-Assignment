import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from model.models import get_model


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Bank ML Dashboard",
    layout="wide",
    page_icon="📊"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.big-metric {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
}
.metric-label {
    font-size: 16px;
    text-align: center;
    color: #9CA3AF;
}
.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
}
.section-title {
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("📊 Bank Marketing ML Dashboard")
st.markdown("### Train and Evaluate Classification Models")

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.markdown("## 📁 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if target_column:

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        model_name = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest"
            ]
        )

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        with st.spinner("Training model..."):
            model = get_model(model_name, X_train)
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # -----------------------------
        # Metrics
        # -----------------------------
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = np.mean(cv_scores)

        st.markdown("## 📈 Model Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Accuracy</div>
            <div class="big-metric">{accuracy:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Precision</div>
            <div class="big-metric">{precision:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Recall</div>
            <div class="big-metric">{recall:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">F1 Score</div>
            <div class="big-metric">{f1:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        col5.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">CV Accuracy</div>
            <div class="big-metric">{cv_mean:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.markdown("## 🧮 Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            linewidths=1,
            linecolor="gray",
            annot_kws={"size": 16, "weight": "bold"}
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # -----------------------------
        # Classification Report
        # -----------------------------
        st.markdown("## 📋 Classification Report")

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.style.format("{:.3f}"),
            use_container_width=True
        )
