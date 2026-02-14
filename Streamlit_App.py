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


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="BITS ML Dashboard", layout="wide")

# -------------------------------------------------
# BITS THEME CSS
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0B1F3A;
}

h1, h2, h3 {
    color: #F5A623;
}

.metric-box {
    background-color: #13294B;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #1E3A8A;
}

.metric-label {
    font-size: 14px;
    color: #AFCBFF;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #F5A623;
}

.stButton>button {
    background-color: #F5A623;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

.stSelectbox label, .stSlider label {
    color: #F5A623 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("📊 BITS Pilani – Machine Learning Dashboard")
st.markdown("Configure model settings and click **Train Model**.")

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Speed optimization
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    # -------------------------------------------------
    # FORM SECTION
    # -------------------------------------------------
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

    # -------------------------------------------------
    # TRAIN AFTER BUTTON
    # -------------------------------------------------
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

        with st.spinner("Training model..."):

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

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        st.markdown("## 📈 Model Performance")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
            "CV Accuracy": cv_scores.mean()
        }

        for col, (label, value) in zip(
            [col1, col2, col3, col4, col5, col6],
            metrics.items()
        ):
            col.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        if auc is not None:
            st.metric("ROC AUC", f"{auc:.4f}")

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------
        st.markdown("## 🧮 Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 18, "weight": "bold"}
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # -------------------------------------------------
        # CLASSIFICATION REPORT
        # -------------------------------------------------
        st.markdown("## 📋 Classification Report")

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)

        st.dataframe(report_df, width="stretch")
