# Indicator.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from models import get_all_models


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(layout="wide")
st.title("🎓 Income Prediction Dashboard")


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

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ==========================================================
# TRAIN MODELS (CACHED)
# ==========================================================
@st.cache_resource
def load_models():

    models = get_all_models(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model in models.values():
        model.fit(X_train, y_train)

    return models


models_dict = load_models()


# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("⚙ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models_dict.keys())
)

uploaded_file = st.sidebar.file_uploader("Upload Test Dataset")

selected_model = models_dict[model_name]


# ==========================================================
# LIVE PREDICTOR
# ==========================================================
st.subheader("🔮 Live Income Predictor")

sample_input = {}

for col in X.columns[:5]:
    if X[col].dtype == "object":
        sample_input[col] = st.selectbox(col, df[col].unique())
    else:
        sample_input[col] = st.number_input(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

if st.button("Predict Income"):
    input_df = pd.DataFrame([sample_input])

    pred = selected_model.predict(input_df)[0]
    prob = selected_model.predict_proba(input_df)[0][1]

    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Income: {label}")
    st.metric(">50K Probability", f"{prob:.2%}")

st.divider()


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
        if st.button("Apply Model"):

            test_df[TARGET] = label_encoder.transform(test_df[TARGET])

            X_test = test_df.drop(columns=[TARGET])
            y_test = test_df[TARGET]

            # Limit KNN to 1000 samples
            if model_name == "KNN" and len(X_test) > 1000:
                X_test = X_test.sample(1000, random_state=42)
                y_test = y_test.loc[X_test.index]

            preds = selected_model.predict(X_test)
            probs = selected_model.predict_proba(X_test)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)
            mcc = matthews_corrcoef(y_test, preds)

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Precision", f"{prec:.3f}")
            col3.metric("Recall", f"{rec:.3f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("F1 Score", f"{f1:.3f}")
            col5.metric("ROC AUC", f"{roc_auc:.3f}")
            col6.metric("MCC", f"{mcc:.3f}")

            # Confusion Matrix
            c1, c2 = st.columns(2)

            with c1:
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(3,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

            with c2:
                fpr, tpr, _ = roc_curve(y_test, probs)
                fig2, ax2 = plt.subplots(figsize=(3,3))
                ax2.plot(fpr, tpr)
                ax2.plot([0,1],[0,1],'--')
                ax2.set_title("ROC Curve")
                st.pyplot(fig2)
