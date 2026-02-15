import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(layout="wide")
st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Pre-trained models • Upload test dataset to evaluate")
st.markdown("---")

# --------------------------------------
# LOAD DATA SAFELY
# --------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

target = "income"

if target not in df.columns:
    st.error("Target column 'income' not found in Data.csv")
    st.stop()

X = df.drop(target, axis=1)
y = df[target]

# Encode target safely
le = LabelEncoder()
y = le.fit_transform(y)

# Separate column types
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# --------------------------------------
# PREPROCESSOR
# --------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# --------------------------------------
# TRAIN MODELS (FAST + SAFE)
# --------------------------------------
@st.cache_resource
def train_models():

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=40,
            max_depth=8,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=40,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1
        )
    }

    trained_models = {}
    results = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():

        if name == "Naive Bayes":
            # GaussianNB cannot handle sparse matrix
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            model.fit(X_train_trans.toarray(), y_train)
            preds = model.predict(X_test_trans.toarray())
            probs = model.predict_proba(X_test_trans.toarray())[:, 1]

            trained_models[name] = (model, True)

        else:
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            probs = pipe.predict_proba(X_test)[:, 1]

            trained_models[name] = (pipe, False)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": roc_auc_score(y_test, probs),
            "MCC": matthews_corrcoef(y_test, preds),
        })

    leaderboard = pd.DataFrame(results).sort_values(
        by="Accuracy", ascending=False
    )

    return trained_models, leaderboard


models, leaderboard_df = train_models()

# --------------------------------------
# TARGET VISUAL
# --------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Target Distribution")
    fig, ax = plt.subplots(figsize=(3, 3))
    df[target].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

with col2:
    st.subheader("🏆 Model Leaderboard")
    st.dataframe(leaderboard_df, use_container_width=True)

st.markdown("---")

# --------------------------------------
# TEST DATA EVALUATION
# --------------------------------------
st.subheader("📂 Evaluate on Test Dataset")

model_choice = st.selectbox("Select Model", leaderboard_df["Model"])

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if target not in test_df.columns:
        st.error("Test file must contain 'income' column.")
    else:

        X_test = test_df.drop(target, axis=1)
        y_test = le.transform(test_df[target])

        model_obj, is_nb = models[model_choice]

        if is_nb:
            X_test_trans = preprocessor.transform(X_test)
            preds = model_obj.predict(X_test_trans.toarray())
            probs = model_obj.predict_proba(X_test_trans.toarray())[:, 1]
        else:
            preds = model_obj.predict(X_test)
            probs = model_obj.predict_proba(X_test)[:, 1]

        st.markdown("---")
        st.subheader("🔮 Classification Performance")

        colA, colB, colC, colD, colE, colF = st.columns(6)

        colA.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
        colB.metric("Precision", f"{precision_score(y_test, preds):.3f}")
        colC.metric("Recall", f"{recall_score(y_test, preds):.3f}")
        colD.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")
        colE.metric("ROC AUC", f"{roc_auc_score(y_test, probs):.3f}")
        colF.metric("MCC", f"{matthews_corrcoef(y_test, preds):.3f}")

        st.markdown("---")

        colX, colY = st.columns(2)

        with colX:
            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

        with colY:
            st.markdown("##### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, probs)
            fig_roc, ax_roc = plt.subplots(figsize=(3, 3))
            ax_roc.plot(fpr, tpr)
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            st.pyplot(fig_roc)
