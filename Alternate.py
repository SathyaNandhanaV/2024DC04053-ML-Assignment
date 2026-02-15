import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(layout="wide")
st.title("🎓 BITS ML Classification Dashboard")

st.markdown("### 🔍 Predict • Evaluate • Understand")

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    df.replace("?", np.nan, inplace=True)
    return df


df = load_data()
target = "income"

X = df.drop(target, axis=1)
y = df[target]

if y.dtype == "object":
    y = y.astype("category").cat.codes

# ----------------------------------
# PREPROCESSOR
# ----------------------------------
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols),
])


# ----------------------------------
# TRAIN MODELS (FAST)
# ----------------------------------
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=10),
        "Decision Tree": DecisionTreeClassifier(max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            verbosity=0
        )
    }

    trained = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        pipe.fit(X, y)
        trained[name] = pipe

    return trained


models_dict = train_models()

# ----------------------------------
# MODEL SELECTOR
# ----------------------------------
selected_model = st.selectbox("Select Model", list(models_dict.keys()))

st.markdown("---")

# ----------------------------------
# TEST DATA UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    if y_test.dtype == "object":
        y_test = y_test.astype("category").cat.codes

    model = models_dict[selected_model]
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # ----------------------------------
    # METRICS
    # ----------------------------------
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    mcc = matthews_corrcoef(y_test, preds)

    # ----------------------------------
    # 🎯 PRIMARY FOCUS — PREDICTION SUMMARY
    # ----------------------------------
    st.header("🔮 Classification Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Class Distribution", 
                  f"Class 1: {(preds==1).sum()} | Class 0: {(preds==0).sum()}")

    with col2:
        st.metric("Accuracy", f"{acc:.4f}")

    st.markdown("---")

    # ----------------------------------
    # 📊 MODEL PERFORMANCE (6 METRICS)
    # ----------------------------------
    st.header("📊 Model Performance")

    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{precision:.4f}")
    m3.metric("Recall", f"{recall:.4f}")
    m4.metric("F1 Score", f"{f1:.4f}")
    m5.metric("ROC AUC", f"{auc:.4f}")
    m6.metric("MCC", f"{mcc:.4f}")

    st.markdown("---")

    # ----------------------------------
    # 🔥 CONFUSION MATRIX
    # ----------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

    # ----------------------------------
    # 📈 ROC CURVE
    # ----------------------------------
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, probs)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    st.pyplot(fig2)
