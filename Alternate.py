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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="BITS ML Classification Dashboard",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (Professional Look)
# -------------------------------------------------
st.markdown("""
<style>
.main {background-color:#0E1117;}
h1,h2,h3,h4 {color:white;}
.metric-box {
    background: #1E222A;
    padding:15px;
    border-radius:12px;
    text-align:center;
}
.big-prediction {
    background: linear-gradient(135deg,#1f77b4,#4fa3ff);
    padding:40px;
    border-radius:16px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------------------------
@st.cache_resource
def load_and_train():

    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()

    target_column = "income"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=400),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            verbosity=0
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC AUC": roc_auc_score(y_test, probs),
            "MCC": matthews_corrcoef(y_test, preds)
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

    return df, results_df, trained_models, le


# -------------------------------------------------
# TRAIN ON STARTUP
# -------------------------------------------------
with st.spinner("Training Pre-Trained Models..."):
    df, leaderboard, models_dict, label_encoder = load_and_train()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🎓 BITS ML Classification Dashboard")
st.caption("Pre-trained models • Upload test dataset to evaluate")

# -------------------------------------------------
# TARGET DISTRIBUTION (Clean Donut)
# -------------------------------------------------
st.subheader("🎯 Target Distribution")

target_counts = df["income"].value_counts()
fig, ax = plt.subplots(figsize=(4,4))
ax.pie(
    target_counts,
    labels=target_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
centre_circle = plt.Circle((0,0),0.60,fc='white')
fig.gca().add_artist(centre_circle)
ax.axis('equal')
st.pyplot(fig)

# -------------------------------------------------
# MODEL LEADERBOARD
# -------------------------------------------------
st.subheader("🏆 Model Performance (Pre-Trained)")

st.dataframe(
    leaderboard.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "ROC AUC": "{:.3f}",
        "MCC": "{:.3f}"
    }),
    use_container_width=True
)

# -------------------------------------------------
# BEST MODEL HIGHLIGHT
# -------------------------------------------------
best_model_name = leaderboard.iloc[0]["Model"]

st.markdown(
    f"""
    <div class="big-prediction">
        Best Performing Model: {best_model_name}
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# TEST DATA SECTION
# -------------------------------------------------
st.divider()
st.subheader("📂 Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV Test File", type="csv")

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    target_column = "income"

    if target_column not in test_df.columns:
        st.error("Test file must contain 'income' column.")
    else:

        X_test = test_df.drop(target_column, axis=1)
        y_test = test_df[target_column]

        X_test = pd.get_dummies(X_test)
        y_test = label_encoder.transform(y_test)

        # Align columns with training
        missing_cols = set(df.drop(target_column,axis=1).columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0

        X_test = X_test.reindex(sorted(X_test.columns), axis=1)

        selected_model = st.selectbox("Select Model for Evaluation", leaderboard["Model"])

        model = models_dict[selected_model]
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1]

        # METRICS
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)
        mcc = matthews_corrcoef(y_test, preds)

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("F1 Score", f"{f1:.3f}")
        col3.metric("ROC AUC", f"{roc:.3f}")
        col4.metric("MCC", f"{mcc:.3f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)

        fig2, ax2 = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)

        # Prediction Distribution
        st.subheader("Prediction Distribution")

        fig3, ax3 = plt.subplots(figsize=(4,3))
        sns.countplot(x=preds, ax=ax3)
        st.pyplot(fig3)
