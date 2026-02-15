import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(layout="wide")

# ==========================================================
# SAFE CSV LOADER
# ==========================================================
def safe_read_csv(file):
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    st.error("Unable to read CSV file.")
    st.stop()

# ==========================================================
# LOAD TRAINING DATA
# ==========================================================
@st.cache_data
def load_data():
    return safe_read_csv("Data.csv")

df = load_data()
df.columns = df.columns.str.strip()

TARGET = "income"

label_encoder = LabelEncoder()
df[TARGET] = label_encoder.fit_transform(df[TARGET])

X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
X = X.astype(float)
y = df[TARGET]
train_columns = X.columns

# ==========================================================
# TRAIN MODELS (CACHED)
# ==========================================================
@st.cache_resource
def train_models():

    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=800
        ),

        "Decision Tree": DecisionTreeClassifier(max_depth=6),

        "KNN": make_pipeline(
            StandardScaler(with_mean=False),
            KNeighborsClassifier(
                n_neighbors=5,
                algorithm="ball_tree",
                leaf_size=40,
                weights="uniform",
                n_jobs=-1
            )
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

models_dict = train_models()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("⚙ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models_dict.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)"
)

selected_model = models_dict[model_name]

# ==========================================================
# HEADER
# ==========================================================
st.title("🎓 Income Classification Dashboard")
st.caption("Pre-trained models • Live Predictor • Stable Evaluation")

# ==========================================================
# 🔮 LIVE INCOME PREDICTOR
# ==========================================================
st.subheader("🔮 Live Income Predictor")

corr = df.corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
top_features = corr.index[1:6]

cols = st.columns(3)
user_input = {}

for i, feature in enumerate(top_features):
    with cols[i % 3]:
        user_input[feature] = st.slider(
            feature,
            int(df[feature].min()),
            int(df[feature].max()),
            int(df[feature].mean())
        )

if st.button("Predict Income"):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df, drop_first=True)

    for col in train_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[train_columns]
    input_df = input_df.astype(float)

    pred = selected_model.predict(input_df)[0]
    prob = selected_model.predict_proba(input_df)[0][1]
    label = label_encoder.inverse_transform([pred])[0]

    c1, c2 = st.columns([2,1])
    c1.success(f"Predicted Income: {label}")
    c2.metric(">50K Probability", f"{prob:.2%}")

st.divider()

# ==========================================================
# 📊 TEST DATA EVALUATION
# ==========================================================
if uploaded_file:

    st.subheader("📊 Test Dataset Evaluation")

    test_df = safe_read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Target column missing in test dataset.")
    else:
        if st.button("Apply Model on Test Data"):

            test_df[TARGET] = label_encoder.transform(test_df[TARGET])

            X_test = pd.get_dummies(test_df.drop(columns=[TARGET]), drop_first=True)

            for col in train_columns:
                if col not in X_test.columns:
                    X_test[col] = 0

            X_test = X_test[train_columns]
            X_test = X_test.astype(float)

            y_test = test_df[TARGET]

            # 🔥 HARD LIMIT FOR KNN
            if model_name == "KNN" and len(X_test) > 1000:
                X_test = X_test.sample(1000, random_state=42)
                y_test = y_test.loc[X_test.index]

            preds = selected_model.predict(X_test)
            probs = selected_model.predict_proba(X_test)[:, 1]

            # ================= METRICS =================
            st.subheader("📊 Test Metrics")

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)

            m1,m2,m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("F1 Score", f"{f1:.3f}")
            m3.metric("ROC AUC", f"{roc_auc:.3f}")

            m4,m5,m6 = st.columns(3)
            m4.metric("Precision", f"{prec:.3f}")
            m5.metric("Recall", f"{rec:.3f}")
            m6.metric("MCC", f"{mcc:.3f}")

            # ================= SMALL PLOTS =================
            colA, colB = st.columns(2)

            with colA:
                cm = confusion_matrix(y_test, preds)
                fig1, ax1 = plt.subplots(figsize=(3,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
                ax1.set_title("Confusion Matrix")
                st.pyplot(fig1)

            with colB:
                fpr, tpr, _ = roc_curve(y_test, probs)
                fig2, ax2 = plt.subplots(figsize=(3,3))
                ax2.plot(fpr, tpr)
                ax2.plot([0,1],[0,1],'--')
                ax2.set_title("ROC Curve")
                st.pyplot(fig2)

            # ================= SUMMARY =================
            st.divider()
            st.subheader("🧠 Model Interpretation")

            st.write(f"""
            Evaluated on **{len(y_test)} samples**.

            This model works by:
            """)

            explanations = {
                "Logistic Regression": "Learning linear relationships using weighted features.",
                "Decision Tree": "Splitting data into rule-based branches.",
                "KNN": "Comparing new samples to nearest stored examples.",
                "Naive Bayes": "Using probabilistic independence assumptions.",
                "Random Forest": "Averaging multiple decision trees.",
                "XGBoost": "Sequentially improving tree errors using boosting."
            }

            st.write(explanations[model_name])

            st.write(f"""
            **Performance Insight**

            Accuracy: {acc:.2%}  
            ROC AUC: {roc_auc:.2f}  
            Precision: {prec:.2f}  
            Recall: {rec:.2f}  
            MCC: {mcc:.2f}
            """)
