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
    confusion_matrix,
    roc_curve
)

from sklearn.model_selection import train_test_split
from model.models import get_model


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="BITS ML Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Pre-trained models | Upload test dataset to evaluate")


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
target_column = "income"

X = df.drop(target_column, axis=1)
y = df[target_column]

if y.dtype == "object":
    y = pd.factorize(y)[0]


# --------------------------------------------------
# PRE-TRAIN MODELS (RUNS ONCE)
# --------------------------------------------------
@st.cache_resource
def train_all_models(X, y):

    trained_models = {}
    results = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_list = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    for model_name in model_list:

        model = get_model(model_name, X_train)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except:
            auc = None

        trained_models[model_name] = model

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "MCC": mcc,
            "ROC AUC": auc if auc else 0
        })

    leaderboard = pd.DataFrame(results).sort_values(
        by="Accuracy", ascending=False
    )

    return trained_models, leaderboard


models, leaderboard = train_all_models(X, y)


# --------------------------------------------------
# BASELINE UI
# --------------------------------------------------
st.header("🏆 Baseline Model Performance")

st.dataframe(leaderboard, use_container_width=True)

best_model_name = leaderboard.iloc[0]["Model"]
st.success(f"🥇 Best Model on Training Split: {best_model_name}")


# --------------------------------------------------
# USER SELECT MODEL
# --------------------------------------------------
st.sidebar.header("🔍 Test Evaluation")

selected_model = st.sidebar.selectbox(
    "Choose Model for Test Evaluation",
    leaderboard["Model"]
)

uploaded_test = st.sidebar.file_uploader(
    "Upload Test CSV",
    type="csv"
)


# --------------------------------------------------
# IF TEST DATA UPLOADED
# --------------------------------------------------
if uploaded_test:

    test_df = pd.read_csv(uploaded_test)
    test_df.columns = test_df.columns.str.strip()

    st.header("📤 Test Dataset Preview")
    st.dataframe(test_df.head(), use_container_width=True)

    if target_column in test_df.columns:

        X_test = test_df.drop(target_column, axis=1)
        y_test = test_df[target_column]

        if y_test.dtype == "object":
            y_test = pd.factorize(y_test)[0]

        model = models[selected_model]
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except:
            auc = None

        # ---------------- METRIC CARDS ----------------
        st.header("📊 Test Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("MCC", f"{mcc:.4f}")

        if auc:
            st.metric("ROC AUC", f"{auc:.4f}")

        # ---------------- CONFUSION MATRIX ----------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            annot_kws={"size": 16, "weight": "bold"}
        )
        st.pyplot(fig)

        # ---------------- CLASSIFICATION REPORT ----------------
        st.subheader("Classification Report")

        report = classification_report(
            y_test, preds, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(4)

        st.dataframe(report_df, use_container_width=True)

        # ---------------- ROC CURVE ----------------
        if auc:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, probs)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, linewidth=3,
                     label=f"AUC = {auc:.3f}")
            ax2.plot([0, 1], [0, 1], linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

    else:
        st.error(
            f"Uploaded test dataset must contain '{target_column}' column."
        )
