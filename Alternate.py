import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="BITS ML Dashboard", layout="wide")

st.title("🎓 BITS ML Classification Dashboard")
st.caption("Pre-trained models • Upload test dataset to evaluate")


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
TARGET = "income"


# ---------------- PRETRAIN MODELS ----------------
@st.cache_resource
def pretrain_models():

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(
            n_estimators=40, max_depth=6, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=-1
        ),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": roc_auc_score(y_test, probs),
            "MCC": matthews_corrcoef(y_test, preds),
        }

        trained[name] = model

    return results, trained, X.columns, le


results, trained_models, train_columns, label_encoder = pretrain_models()


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(trained_models.keys())
)

uploaded = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type="csv"
)


# ---------------- MAIN LAYOUT ----------------
left_col, right_col = st.columns([1, 2])


# ================= LEFT SIDE =================
with left_col:

    st.subheader("🎯 Target Distribution")

    counts = df[TARGET].value_counts()

    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_title("Income Distribution")
    st.pyplot(fig)


# ================= RIGHT SIDE =================
with right_col:

    st.subheader("🏆 Pre-Trained Model Comparison")

    perf_df = pd.DataFrame(results).T.round(4)
    st.dataframe(perf_df, use_container_width=True)


# ---------------- TEST EVALUATION ----------------
if uploaded:

    st.divider()
    st.header("🔎 Test Dataset Evaluation")

    test_df = pd.read_csv(uploaded)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Test data must contain 'income' column.")
        st.stop()

    X_test = test_df.drop(TARGET, axis=1)
    y_test = label_encoder.transform(test_df[TARGET])

    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=train_columns, fill_value=0)

    model = trained_models[model_name]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)
    mcc = matthews_corrcoef(y_test, preds)

    # -------- Metrics --------
    st.subheader("📊 Model Performance (Test Data)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1 Score", f"{f1:.4f}")
    c5.metric("ROC AUC", f"{roc:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # -------- Confusion Matrix --------
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test, preds)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax2
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # -------- Summary --------
    st.subheader("📝 Model Summary")

    st.info(
        f"""
        The selected model **{model_name}** achieved an accuracy of **{acc:.2%}**
        on the uploaded test dataset.

        • F1 Score indicates balance between precision and recall.
        • ROC AUC measures class separability.
        • MCC reflects overall prediction quality including imbalance.

        This model appears to {'perform well' if acc > 0.8 else 'require improvement'} 
        on the provided dataset.
        """
    )
