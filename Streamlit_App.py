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


# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="ML Classification Dashboard", layout="wide")
st.title("🚀 ML Classification Dashboard")


uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.replace("?", np.nan, inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Sparse Preprocessor
    preprocessor_sparse = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    # Dense for Naive Bayes
    preprocessor_dense = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])

    model_option = st.selectbox(
        "🤖 Select Model",
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

    if model_option == "Naive Bayes":
        selected_preprocessor = preprocessor_dense
    else:
        selected_preprocessor = preprocessor_sparse

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", selected_preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    # -----------------------
    # METRICS
    # -----------------------
    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📈 Graphs", "🌲 Feature Importance"])

    # ===============================
    # TAB 1 — METRICS
    # ===============================
    with tab1:

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))

    # ===============================
    # TAB 2 — GRAPHS
    # ===============================
    with tab2:

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
        st.pyplot(fig1)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr)
        ax2.plot([0, 1], [0, 1])
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        st.pyplot(fig2)

        st.subheader("Metrics Comparison")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            "Value": [accuracy, auc, precision, recall, f1, mcc]
        })

        fig3, ax3 = plt.subplots()
        sns.barplot(data=metrics_df, x="Metric", y="Value", ax=ax3)
        st.pyplot(fig3)

    # ===============================
    # TAB 3 — FEATURE IMPORTANCE
    # ===============================
    with tab3:

        if model_option in ["Random Forest", "XGBoost"]:

            model_fitted = pipeline.named_steps["classifier"]
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

            importances = model_fitted.feature_importances_

            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)

            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax4)
            st.pyplot(fig4)

        else:
            st.info("Feature importance available only for Random Forest and XGBoost.")
