import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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


# ---------------- PRE-TRAIN MODELS ----------------
@st.cache_resource
def train_models():

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # One hot encoding
    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, max_depth=8, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probs)
        else:
            roc = 0

        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": roc,
            "MCC": matthews_corrcoef(y_test, preds),
        }

        trained_models[name] = model

    return results, trained_models, X.columns, le


results, trained_models, training_columns, label_encoder = train_models()


# ---------------- TARGET DISTRIBUTION (SMALL BAR CHART) ----------------
st.subheader("📊 Target Distribution")

class_counts = df[TARGET].value_counts()

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(class_counts.index, class_counts.values)
ax.set_ylabel("Count")
ax.set_xlabel("Class")
ax.set_title("Income Class Distribution")
st.pyplot(fig)


# ---------------- MODEL PERFORMANCE ----------------
st.subheader("🏆 Model Performance (Pre-Trained)")

results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
results_df = results_df.round(4)

st.dataframe(results_df, use_container_width=True)


# ---------------- SELECT MODEL ----------------
st.subheader("🔎 Evaluate on Test Data")

selected_model = st.selectbox(
    "Select Model",
    list(trained_models.keys())
)


# ---------------- TEST DATA UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Test CSV", type="csv")

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Test data must contain 'income' column.")
        st.stop()

    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]

    X_test = pd.get_dummies(X_test, drop_first=True)

    # Align columns
    X_test = X_test.reindex(columns=training_columns, fill_value=0)

    y_test_encoded = label_encoder.transform(y_test)

    model = trained_models[selected_model]

    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test_encoded, probs)
    else:
        roc = 0

    acc = accuracy_score(y_test_encoded, preds)
    prec = precision_score(y_test_encoded, preds)
    rec = recall_score(y_test_encoded, preds)
    f1 = f1_score(y_test_encoded, preds)
    mcc = matthews_corrcoef(y_test_encoded, preds)

    st.subheader("📈 Test Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("ROC AUC", f"{roc:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test_encoded, preds)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.imshow(cm)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)
