import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

# ==========================================================
# SAFE CSV READER (FIXES UnicodeDecodeError)
# ==========================================================
def safe_read_csv(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin1")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = safe_read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
TARGET = "income"

# Encode target
label_encoder = LabelEncoder()
df[TARGET] = label_encoder.fit_transform(df[TARGET])

# One-hot encode (reduced size)
X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
X = X.astype(float)
y = df[TARGET]
train_columns = X.columns

# ==========================================================
# TRAIN MODELS (STABLE + FAST)
# ==========================================================
@st.cache_resource
def train_models():

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=800,
            solver="liblinear"
        ),

        "Decision Tree": DecisionTreeClassifier(max_depth=6),

        "KNN": make_pipeline(
            StandardScaler(with_mean=False),
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance"
            )
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "mcc": matthews_corrcoef(y_test, preds)
        }

    return results

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

selected_model = models_dict[model_name]["model"]

# ==========================================================
# HEADER
# ==========================================================
st.title("🎓 BITS ML Classification Dashboard")
st.caption("Pre-trained models • Live predictor • Upload test dataset")

# ==========================================================
# LIVE INCOME PREDICTOR
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
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=train_columns, fill_value=0)
    input_df = input_df.astype(float)

    pred = selected_model.predict(input_df)[0]
    prob = selected_model.predict_proba(input_df)[0][1]
    label = label_encoder.inverse_transform([pred])[0]

    c1, c2 = st.columns([2,1])
    c1.success(f"Predicted Income: {label}")
    c2.metric(">50K Probability", f"{prob:.2%}")

# ==========================================================
# MODEL COMPARISON TABLE
# ==========================================================
st.subheader("🏆 Pre-Trained Model Comparison")

table_data = []
for name, v in models_dict.items():
    table_data.append([
        name, v["accuracy"], v["precision"],
        v["recall"], v["f1"], v["roc_auc"], v["mcc"]
    ])

results_df = pd.DataFrame(
    table_data,
    columns=["Model","Accuracy","Precision","Recall","F1","ROC AUC","MCC"]
).sort_values("Accuracy", ascending=False)

numeric_cols = ["Accuracy","Precision","Recall","F1","ROC AUC","MCC"]

st.dataframe(
    results_df.style
        .format({c: "{:.3f}" for c in numeric_cols})
        .background_gradient(cmap="Blues", subset=numeric_cols),
    width="stretch"
)

# ==========================================================
# TEST DATA EVALUATION
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
            X_test = X_test.reindex(columns=train_columns, fill_value=0)
            X_test = X_test.astype(float)
            y_test = test_df[TARGET]

            preds = selected_model.predict(X_test)
            probs = selected_model.predict_proba(X_test)[:,1]

            # METRICS FIRST
            st.subheader("📊 Test Metrics")

            acc = accuracy_score(y_test,preds)
            f1 = f1_score(y_test,preds)
            roc_auc = roc_auc_score(y_test,probs)
            prec = precision_score(y_test,preds)
            rec = recall_score(y_test,preds)
            mcc = matthews_corrcoef(y_test,preds)

            m1,m2,m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("F1 Score", f"{f1:.3f}")
            m3.metric("ROC AUC", f"{roc_auc:.3f}")

            m4,m5,m6 = st.columns(3)
            m4.metric("Precision", f"{prec:.3f}")
            m5.metric("Recall", f"{rec:.3f}")
            m6.metric("MCC", f"{mcc:.3f}")

            # SIDE BY SIDE PLOTS
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

            st.success("Evaluation completed successfully.")
