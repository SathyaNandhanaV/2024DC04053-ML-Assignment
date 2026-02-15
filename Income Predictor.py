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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

TARGET = "income"

# -----------------------------
# PREPROCESS
# -----------------------------
@st.cache_resource
def preprocess_and_train():

    data = df.copy()

    le = LabelEncoder()
    data[TARGET] = le.fit_transform(data[TARGET])

    X = data.drop(TARGET, axis=1)
    y = data[TARGET]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=8),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=60,
            max_depth=5,
            learning_rate=0.1
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": roc_auc_score(y_test, probs),
            "MCC": matthews_corrcoef(y_test, preds),
        }

    return results, X.columns, le


models_dict, train_columns, label_encoder = preprocess_and_train()

# -----------------------------
# HEADER
# -----------------------------
st.title("🎓 BITS ML Classification Dashboard")
st.caption("Pre-trained models • Upload test dataset • Live simulator")

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns([1, 3])

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    st.subheader("⚙ Configuration")

    model_name = st.selectbox(
        "Select Model",
        list(models_dict.keys())
    )

    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type="csv"
    )

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:

    # -------------------------
    # TARGET DISTRIBUTION
    # -------------------------
    st.subheader("🎯 Target Distribution")

    counts = df[TARGET].value_counts()

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(counts.index, counts.values)
    ax.set_ylabel("Count")
    ax.set_title("Income Distribution")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height}",
            ha='center',
            va='bottom'
        )

    st.pyplot(fig)

    # -------------------------
    # MODEL LEADERBOARD
    # -------------------------
    st.subheader("🏆 Pre-Trained Model Comparison")

    leaderboard = pd.DataFrame(models_dict).T
    leaderboard = leaderboard.drop(columns=["model"])
    leaderboard = leaderboard.sort_values("Accuracy", ascending=False)

    styled = leaderboard.style.format("{:.3f}").background_gradient(cmap="Blues")

    st.dataframe(styled, use_container_width=True)

# -----------------------------
# TEST DATA EVALUATION
# -----------------------------
if uploaded_file is not None:

    st.divider()
    st.header("📊 Test Dataset Evaluation")

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Target column missing in test dataset.")
    else:

        test_df[TARGET] = label_encoder.transform(test_df[TARGET])

        X_test_user = test_df.drop(TARGET, axis=1)
        y_test_user = test_df[TARGET]

        X_test_user = pd.get_dummies(X_test_user)
        X_test_user = X_test_user.reindex(columns=train_columns, fill_value=0)

        model = models_dict[model_name]["model"]

        preds = model.predict(X_test_user)
        probs = model.predict_proba(X_test_user)[:, 1]

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y_test_user, preds):.3f}")
        col2.metric("F1 Score", f"{f1_score(y_test_user, preds):.3f}")
        col3.metric("ROC AUC", f"{roc_auc_score(y_test_user, probs):.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{precision_score(y_test_user, preds):.3f}")
        col5.metric("Recall", f"{recall_score(y_test_user, preds):.3f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_test_user, preds):.3f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test_user, preds)

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        st.pyplot(fig2)

        st.success(
            f"Model '{model_name}' evaluated successfully on uploaded dataset."
        )

# -----------------------------
# LIVE SIMULATOR
# -----------------------------
st.divider()
st.header("🔮 Live Income Prediction Simulator")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    hours = st.slider("Hours per Week", 1, 80, 40)

with col2:
    education = st.selectbox("Education", df["education"].unique())
    gender = st.selectbox("Gender", df["gender"].unique())

if st.button("🚀 Predict Income"):

    user_input = pd.DataFrame([{
        "age": age,
        "hours-per-week": hours,
        "education": education,
        "gender": gender
    }])

    user_input = pd.get_dummies(user_input)
    user_input = user_input.reindex(columns=train_columns, fill_value=0)

    model = models_dict[model_name]["model"]

    pred = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]

    income_label = label_encoder.inverse_transform([pred])[0]

    st.subheader("Prediction Result")

    if income_label == ">50K":
        st.success(f"Predicted Income: {income_label}")
    else:
        st.info(f"Predicted Income: {income_label}")

    st.metric("Probability of >50K", f"{prob:.2%}")
