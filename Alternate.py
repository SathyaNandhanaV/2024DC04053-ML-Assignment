import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Pre-trained models • Upload test dataset to evaluate")
st.markdown("---")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

target = "income"

X = df.drop(target, axis=1)
y = df[target]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Identify columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# --------------------------------------------------
# PRETRAIN MODELS (FAST SETTINGS)
# --------------------------------------------------
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": roc_auc_score(y_test, probs),
            "MCC": matthews_corrcoef(y_test, preds)
        })

        trained_models[name] = pipe

    leaderboard = pd.DataFrame(results).sort_values(
        by="Accuracy", ascending=False
    )

    return trained_models, leaderboard


models, leaderboard_df = train_models()

# --------------------------------------------------
# TARGET VISUALIZATION
# --------------------------------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("🎯 Target Distribution")
    fig, ax = plt.subplots(figsize=(3,3))
    df[target].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Pre-trained Leaderboard")
    st.dataframe(leaderboard_df, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# TEST DATA EVALUATION
# --------------------------------------------------
st.subheader("📂 Evaluate Model on Test Dataset")

model_choice = st.selectbox(
    "Select Model",
    leaderboard_df["Model"]
)

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if target not in test_df.columns:
        st.error(f"Test file must contain '{target}' column.")
    else:
        X_test = test_df.drop(target, axis=1)
        y_test = le.transform(test_df[target])

        selected_model = models[model_choice]

        preds = selected_model.predict(X_test)
        probs = selected_model.predict_proba(X_test)[:,1]

        st.markdown("---")
        st.subheader("🔮 Classification Result")

        acc = accuracy_score(y_test, preds)

        st.metric("Accuracy", f"{acc:.4f}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precision", f"{precision_score(y_test, preds):.3f}")
        m2.metric("Recall", f"{recall_score(y_test, preds):.3f}")
        m3.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")
        m4.metric("ROC AUC", f"{roc_auc_score(y_test, probs):.3f}")
        m5.metric("MCC", f"{matthews_corrcoef(y_test, preds):.3f}")

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(3,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

        with c2:
            st.markdown("##### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, probs)
            fig_roc, ax_roc = plt.subplots(figsize=(3,3))
            ax_roc.plot(fpr, tpr)
            ax_roc.plot([0,1],[0,1], linestyle="--")
            st.pyplot(fig_roc)
