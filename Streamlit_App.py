import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("📊 ML Classification Models Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace("?", np.nan, inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    model_option = st.selectbox(
        "Select Model",
        ["Logistic Regression",
         "Decision Tree",
         "KNN",
         "Naive Bayes",
         "Random Forest",
         "XGBoost"]
    )

    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    model = model_dict[model_option]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    # ======================
    # METRICS
    # ======================
    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    st.subheader("📌 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ======================
    # CONFUSION MATRIX
    # ======================
    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, preds)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    # ======================
    # ROC CURVE
    # ======================
    st.subheader("📈 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, probs)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    st.pyplot(fig2)

    # ======================
    # METRICS BAR CHART
    # ======================
    st.subheader("📊 Metrics Comparison")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        "Value": [accuracy, auc, precision, recall, f1, mcc]
    })

    fig3, ax3 = plt.subplots()
    sns.barplot(data=metrics_df, x="Metric", y="Value", ax=ax3)
    st.pyplot(fig3)

    # ======================
    # CLASS DISTRIBUTION
    # ======================
    st.subheader("📌 Class Distribution")

    fig4, ax4 = plt.subplots()
    y.value_counts().plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Count")
    st.pyplot(fig4)
