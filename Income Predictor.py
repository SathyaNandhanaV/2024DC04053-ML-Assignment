import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder

from models import get_all_models  # your existing model loader

st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
TARGET = "income"

# -------------------------------
# PREPROCESS
# -------------------------------
label_encoder = LabelEncoder()
df[TARGET] = label_encoder.fit_transform(df[TARGET])

X = pd.get_dummies(df.drop(columns=[TARGET]))
y = df[TARGET]

train_columns = X.columns

# -------------------------------
# TRAIN MODELS ONCE
# -------------------------------
@st.cache_resource
def train_models():
    models = get_all_models()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

# ===============================
# HEADER
# ===============================
st.title("🎓 BITS ML Classification Dashboard")
st.caption("Pre-trained models • Upload test dataset to evaluate")

# ===============================
# 🎯 LIVE PREDICTOR AT TOP
# ===============================
st.divider()
st.header("🎯 Live Income Predictor")

numeric_df = df.copy()
corr = numeric_df.corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
top_features = corr.index[1:7]

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

model_name = st.selectbox("Select Model", list(models_dict.keys()))
selected_model = models_dict[model_name]["model"]

if st.button("🔮 Predict Income"):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    pred = selected_model.predict(input_df)[0]
    prob = selected_model.predict_proba(input_df)[0][1]

    label = label_encoder.inverse_transform([pred])[0]

    col1, col2 = st.columns([2,1])

    with col1:
        st.success(f"Predicted Income: {label}")

    with col2:
        st.metric(">50K Probability", f"{prob:.2%}")

    # Small probability chart
    fig, ax = plt.subplots(figsize=(3,2))
    bars = ax.bar(["<=50K", ">50K"], [1-prob, prob])
    ax.set_ylim(0,1)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, height/2,
                f"{height:.2%}", ha='center', va='center', color='white', fontsize=8)
    ax.set_title("Confidence", fontsize=9)
    st.pyplot(fig)

# ===============================
# 📊 PRETRAINED MODEL TABLE
# ===============================
st.divider()
st.header("🏆 Pre-Trained Model Comparison")

table_data = []
for name, values in models_dict.items():
    table_data.append([
        name,
        values["accuracy"],
        values["precision"],
        values["recall"],
        values["f1"],
        values["roc_auc"],
        values["mcc"]
    ])

results_df = pd.DataFrame(
    table_data,
    columns=["Model","Accuracy","Precision","Recall","F1","ROC AUC","MCC"]
).sort_values("Accuracy", ascending=False)

styled_table = (
    results_df.style
    .background_gradient(cmap="Blues", subset=["Accuracy","F1","ROC AUC"])
    .format("{:.3f}", subset=["Accuracy","Precision","Recall","F1","ROC AUC","MCC"])
)

st.dataframe(styled_table, use_container_width=True)

# ===============================
# 📂 TEST DATASET SECTION
# ===============================
st.divider()
st.header("📂 Evaluate Uploaded Test Dataset")

uploaded = st.file_uploader("Upload CSV Test Data")

if uploaded:
    test_df = pd.read_csv(uploaded)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Target column missing in test dataset.")
    else:
        if st.button("Apply Model on Test Data"):
            test_df[TARGET] = label_encoder.transform(test_df[TARGET])

            X_test = pd.get_dummies(test_df.drop(columns=[TARGET]))
            X_test = X_test.reindex(columns=train_columns, fill_value=0)
            y_test = test_df[TARGET]

            preds = selected_model.predict(X_test)
            probs = selected_model.predict_proba(X_test)[:,1]

            # Small confusion matrix
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(3,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix", fontsize=10)
            st.pyplot(fig_cm)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, probs)
            fig_roc, ax_roc = plt.subplots(figsize=(3,3))
            ax_roc.plot(fpr, tpr)
            ax_roc.plot([0,1],[0,1],'--')
            ax_roc.set_title("ROC Curve", fontsize=10)
            st.pyplot(fig_roc)

            # Metrics
            st.subheader("📊 Test Metrics")
            m1, m2, m3 = st.columns(3)

            m1.metric("Accuracy", f"{accuracy_score(y_test,preds):.3f}")
            m2.metric("F1 Score", f"{f1_score(y_test,preds):.3f}")
            m3.metric("ROC AUC", f"{roc_auc_score(y_test,probs):.3f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("Precision", f"{precision_score(y_test,preds):.3f}")
            m5.metric("Recall", f"{recall_score(y_test,preds):.3f}")
            m6.metric("MCC", f"{matthews_corrcoef(y_test,preds):.3f}")

            st.success("Evaluation Complete")
