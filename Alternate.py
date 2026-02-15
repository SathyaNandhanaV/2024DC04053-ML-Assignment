import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from model.models import get_model

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="BITS ML Dashboard",
    layout="wide"
)

# ------------------ HEADER ------------------ #
st.markdown("""
<h1 style='font-size:40px;'>🎓 BITS ML Classification Dashboard</h1>
<p style='color:gray;'>Pre-trained models • Upload test dataset to evaluate</p>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------ #
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ------------------ TARGET DISTRIBUTION ------------------ #
st.markdown("## 🎯 Target Distribution")

col1, col2 = st.columns([1, 2])

target_column = "income"

target_counts = df[target_column].value_counts()
target_percent = target_counts / len(df) * 100

# ----- Donut Chart ----- #
with col1:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        target_counts,
        labels=target_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={'width':0.4}
    )
    ax.set_title("Class Split")
    st.pyplot(fig)

# ------------------ TRAIN MODELS ------------------ #
@st.cache_resource
def train_models():
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    results = []

    for name in model_names:
        model = get_model(name, X_train)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(
                (y_test == y_test.unique()[1]).astype(int),
                probs
            )
        else:
            auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC AUC": auc,
            "MCC": mcc
        })

    return pd.DataFrame(results).sort_values("Accuracy", ascending=False)

leaderboard = train_models()

# ------------------ MODEL LEADERBOARD ------------------ #
st.markdown("## 🏆 Model Leaderboard (Pre-Trained)")

styled_lb = leaderboard.style.format({
    "Accuracy": "{:.3f}",
    "Precision": "{:.3f}",
    "Recall": "{:.3f}",
    "F1": "{:.3f}",
    "ROC AUC": "{:.3f}",
    "MCC": "{:.3f}"
}).background_gradient(cmap="Blues", subset=["Accuracy"])

st.dataframe(styled_lb, use_container_width=True)

# ------------------ TEST DATA SECTION ------------------ #
st.markdown("## 📂 Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    model_choice = st.selectbox(
        "Select Model",
        leaderboard["Model"].tolist()
    )

    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=pd.get_dummies(df.drop(target_column, axis=1)).columns, fill_value=0)

    model = get_model(model_choice, X_test)
    model.fit(pd.get_dummies(df.drop(target_column, axis=1)), df[target_column])

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")
    mcc = matthews_corrcoef(y_test, preds)

    # ------------------ RESULTS DISPLAY ------------------ #
    st.markdown("## 🔍 Prediction Summary")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1 Score", f"{f1:.3f}")
    m5.metric("MCC", f"{mcc:.3f}")

    # ------------------ CONFUSION MATRIX ------------------ #
    st.markdown("### 🔢 Confusion Matrix")

    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ------------------ CLASSIFICATION REPORT ------------------ #
    st.markdown("### 📊 Detailed Classification Report")

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
