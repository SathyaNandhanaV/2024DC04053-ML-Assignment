import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
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


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Pre-trained models • Upload test dataset to evaluate")


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
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


# ------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------
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


# ------------------------------------------------
# TRAIN MODELS (FAST VERSION)
# ------------------------------------------------
@st.cache_resource
def train_models():

    # Speed optimization: sample max 20k rows
    if len(X) > 20000:
        X_sample = X.sample(n=20000, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=8, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1
        )
    }

    trained = {}
    leaderboard = []

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        try:
            probs = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0

        leaderboard.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": auc,
            "MCC": matthews_corrcoef(y_test, preds)
        })

        trained[name] = pipe

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        by="Accuracy", ascending=False
    )

    return trained, leaderboard_df


models_dict, leaderboard_df = train_models()


# ------------------------------------------------
# TARGET VARIABLE VISUAL
# ------------------------------------------------
st.markdown("## 🎯 Target Variable Distribution")

fig, ax = plt.subplots(figsize=(4,4))
y.value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=["#1f77b4", "#ff7f0e"],
    ax=ax
)
ax.set_ylabel("")
ax.set_title("Income Classification Distribution")
st.pyplot(fig)


# ------------------------------------------------
# PRE-TRAINED MODEL HIGHLIGHT
# ------------------------------------------------
st.markdown("## 🏆 Pre-Trained Model Performance")

best_model = leaderboard_df.iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("🥇 Best Model", best_model["Model"])
col2.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
col3.metric("ROC AUC", f"{best_model['ROC AUC']:.4f}")

# SAFE STYLING (NO CRASH)
numeric_cols = leaderboard_df.select_dtypes(include="number").columns

styled = leaderboard_df.style \
    .format({col: "{:.4f}" for col in numeric_cols}) \
    .background_gradient(subset=numeric_cols, cmap="viridis")

st.dataframe(styled, use_container_width=True)


# ------------------------------------------------
# UPLOAD TEST DATA
# ------------------------------------------------
st.markdown("## 📂 Upload Test Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    if y_test.dtype == "object":
        y_test = y_test.astype("category").cat.codes

    selected_model = st.selectbox(
        "Select Model for Evaluation",
        list(models_dict.keys())
    )

    model = models_dict[selected_model]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # CLASSIFICATION RESULTS
    # -----------------------------
    st.markdown("## 🔮 Classification Results")

    colA, colB = st.columns(2)
    colA.metric("🎯 Accuracy", f"{accuracy_score(y_test, preds):.4f}")
    colB.metric("📊 F1 Score", f"{f1_score(y_test, preds):.4f}")

    # Prediction distribution
    st.markdown("### Prediction Distribution")

    fig_pred, ax_pred = plt.subplots(figsize=(4,4))
    pd.Series(preds).value_counts().plot.pie(
        autopct="%1.1f%%",
        colors=["#2ca02c", "#d62728"],
        ax=ax_pred
    )
    ax_pred.set_ylabel("")
    ax_pred.set_title("Predicted Class Distribution")
    st.pyplot(fig_pred)

    # Detailed Metrics
    st.markdown("### 📊 Detailed Metrics")

    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    m1.metric("Precision", f"{precision_score(y_test, preds):.4f}")
    m2.metric("Recall", f"{recall_score(y_test, preds):.4f}")
    m3.metric("F1 Score", f"{f1_score(y_test, preds):.4f}")
    m4.metric("ROC AUC", f"{roc_auc_score(y_test, probs):.4f}")
    m5.metric("MCC", f"{matthews_corrcoef(y_test, preds):.4f}")
    m6.metric("Support", len(y_test))

    # Confusion Matrix
    st.markdown("### 🔥 Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    # ROC Curve
    st.markdown("### 📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig_roc, ax_roc = plt.subplots(figsize=(5,4))
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0,1],[0,1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    st.pyplot(fig_roc)
