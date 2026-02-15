#Name: SATHYA NANDHANA V
#BITS ID: 2024DC04053

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from sklearn.model_selection import train_test_split
from model.models import get_all_models

st.set_page_config(layout="wide")


@st.cache_resource
def load_models():
    return get_all_models()

models_dict, train_columns, label_encoder = load_models()


st.sidebar.title("⚙ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models_dict.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)"
)

selected_model = models_dict[model_name]


st.title("🎓 Income Predictor")
st.caption("Pre-trained models • Live Prediction • Test Evaluation")


st.subheader("🔮 Live Income Predictor")

df = pd.read_csv("Data.csv")
df.columns = df.columns.str.strip()

TARGET = "income"
df[TARGET] = label_encoder.transform(df[TARGET])

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

    input_df = input_df[train_columns].astype(float)

    pred = selected_model.predict(input_df)[0]
    prob = selected_model.predict_proba(input_df)[0][1]
    label = label_encoder.inverse_transform([pred])[0]

    c1, c2 = st.columns([2,1])
    c1.success(f"Predicted Income: {label}")
    c2.metric(">50K Probability", f"{prob:.2%}")

st.divider()


@st.cache_data
def compute_pretrained_metrics():

    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()

    TARGET = "income"
    df[TARGET] = label_encoder.transform(df[TARGET])

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
    X = X.astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    for name, model in models_dict.items():
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

    return pd.DataFrame(results)


st.subheader("🏆 Pre-Trained Model Performance")

metrics_df = compute_pretrained_metrics()

styled_df = (
    metrics_df
    .style
    .background_gradient(cmap="Blues", subset=metrics_df.columns[1:])
    .format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
        "ROC AUC": "{:.3f}",
        "MCC": "{:.3f}"
    })
)

st.dataframe(styled_df, width="stretch")

st.divider()


if uploaded_file:

    st.subheader("📊 Test Dataset Evaluation")

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Target column missing in test dataset.")
    else:

        if st.button("Apply Model on Test Data"):

            test_df[TARGET] = label_encoder.transform(test_df[TARGET])

            X_test = pd.get_dummies(
                test_df.drop(columns=[TARGET]),
                drop_first=True
            )

            for col in train_columns:
                if col not in X_test.columns:
                    X_test[col] = 0

            X_test = X_test[train_columns].astype(float)
            y_test = test_df[TARGET]

            if model_name == "KNN" and len(X_test) > 1000:
                X_test = X_test.sample(1000, random_state=42)
                y_test = y_test.loc[X_test.index]

            preds = selected_model.predict(X_test)
            probs = selected_model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)

            st.subheader("📊 Test Metrics")

            m1,m2,m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("F1 Score", f"{f1:.3f}")
            m3.metric("ROC AUC", f"{roc_auc:.3f}")

            m4,m5,m6 = st.columns(3)
            m4.metric("Precision", f"{prec:.3f}")
            m5.metric("Recall", f"{rec:.3f}")
            m6.metric("MCC", f"{mcc:.3f}")

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

            
            st.divider()
            st.subheader("🧠 Detailed Performance Analysis")

            total = len(y_test)
            positive = (y_test == 1).sum()
            negative = (y_test == 0).sum()
            imbalance = round((positive / total) * 100, 2)

            st.markdown(f"""
### 📊 Dataset Insights

The uploaded test dataset contains **{total} records**.

- **≤50K class:** {negative} instances  
- **>50K class:** {positive} instances  
- High-income representation: **{imbalance}%**

This indicates the dataset is {'balanced' if 40 < imbalance < 60 else 'moderately imbalanced'}.
""")

            st.markdown(f"""
### 📈 Model Evaluation Breakdown

**Accuracy ({acc:.2%})**  
Represents overall correctness. The model correctly classifies income levels in most cases.

**Precision ({prec:.2f})**  
Indicates how reliable high-income predictions are. Higher precision means fewer false positives.

**Recall ({rec:.2f})**  
Shows how effectively the model captures actual high-income individuals.

**F1 Score ({f1:.2f})**  
Balances precision and recall. Useful when classes are not perfectly balanced.

**ROC AUC ({roc_auc:.2f})**  
Measures the model's ability to separate income classes across all thresholds.

**MCC ({mcc:.2f})**  
A robust metric that evaluates balanced classification performance,
especially valuable when datasets are imbalanced.
""")

            if acc > 0.85 and mcc > 0.6:
                st.success("Overall, the model demonstrates strong generalization and reliable predictive performance.")
            elif acc > 0.75:
                st.info("The model performs reasonably well but could benefit from further optimization.")
            else:
                st.warning("Model performance suggests potential underfitting or feature limitations.")
