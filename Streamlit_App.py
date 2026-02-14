import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="2024DC04053 ML Assignment-1",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# BITS THEME STYLING
# ---------------------------------------------------
st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color:#003366;}
h3 {color:#d4af37;}
.stMetric {
    background-color:white;
    padding:10px;
    border-radius:10px;
    border-left:5px solid #d4af37;
}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <h1>BITS Pilani</h1>
    <h3>Work Integrated Learning Programme</h3>
    <h2>Machine Learning Classification Dashboard</h2>
</div>
<hr>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🎓 Dashboard Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

cv_folds = st.sidebar.slider("Cross Validation Folds", 3, 10, 5)

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.replace("?", np.nan, inplace=True)

    target_column = st.sidebar.selectbox("Select Target Column", df.columns)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    results = []

    st.subheader("🚀 Training All Models...")

    for name, model in models.items():

        preprocessor = preprocessor_dense if name == "Naive Bayes" else preprocessor_sparse

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        auc = roc_auc_score(y_test, probs) if probs is not None else 0

        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="accuracy")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc,
            "CV Mean Accuracy": cv_scores.mean()
        })

    results_df = pd.DataFrame(results).sort_values(
        by="CV Mean Accuracy", ascending=False
    )

    # ---------------------------------------------------
    # LEADERBOARD
    # ---------------------------------------------------
    st.subheader("🏆 Multi-Model Leaderboard (Ranked by CV Accuracy)")
    st.dataframe(results_df, use_container_width=True)

    # ---------------------------------------------------
    # BAR CHART COMPARISON
    # ---------------------------------------------------
    st.subheader("📊 Model Performance Comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=results_df,
        x="CV Mean Accuracy",
        y="Model",
        ax=ax
    )
    ax.set_title("Cross-Validation Accuracy Comparison")
    st.pyplot(fig)

    # ---------------------------------------------------
    # BEST MODEL DETAILS
    # ---------------------------------------------------
    best_model_name = results_df.iloc[0]["Model"]
    st.subheader(f"🥇 Best Model: {best_model_name}")

    best_model = models[best_model_name]
    best_preprocessor = preprocessor_dense if best_model_name == "Naive Bayes" else preprocessor_sparse

    best_pipeline = Pipeline([
        ("preprocessor", best_preprocessor),
        ("classifier", best_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_pipeline.fit(X_train, y_train)
    preds = best_pipeline.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title("Confusion Matrix (Best Model)")
    st.pyplot(fig2)

else:
    st.info("👈 Upload a dataset to begin.")
