import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="BITS ML Predictor", layout="wide")

st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Preloaded training data. Upload only test data to evaluate.")

# ---------------- LOAD TRAINING DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------------- DATA EXPLORATION ----------------
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

with col2:
    st.write("Missing Values")
    st.write(df.isnull().sum())

# Target
target_column = "income"

# Encode target
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column])

X = df.drop(target_column, axis=1)
y = df[target_column]

# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
            "mcc": matthews_corrcoef(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
            "cv": cross_val_score(model, X, y, cv=3).mean(),
            "X_test": X_test,
            "y_test": y_test
        }

    return results

models = train_models()

# ---------------- MODEL LEADERBOARD ----------------
st.header("🏆 Model Leaderboard")

leaderboard = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": data["accuracy"],
        "F1 Score": data["f1"],
        "ROC AUC": data["auc"],
        "CV Score": data["cv"]
    }
    for name, data in models.items()
]).sort_values(by="Accuracy", ascending=False)

st.dataframe(leaderboard, use_container_width=True)

best_model_name = leaderboard.iloc[0]["Model"]
best_model = models[best_model_name]["model"]

st.success(f"🥇 Best Model: {best_model_name}")

# ---------------- BASELINE EVALUATION ----------------
st.header("📈 Baseline Evaluation (Built-in Test Split)")

y_test = models[best_model_name]["y_test"]
X_test = models[best_model_name]["X_test"]
preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
col2.metric("Precision", f"{precision_score(y_test, preds):.3f}")
col3.metric("Recall", f"{recall_score(y_test, preds):.3f}")
col4.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, probs)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, probs):.3f}")
ax2.plot([0,1],[0,1],'--')
ax2.legend()
st.pyplot(fig2)

# ---------------- TEST DATA UPLOAD ----------------
st.header("📤 Upload Test Dataset")

uploaded_test = st.file_uploader("Upload Test CSV", type="csv")

if uploaded_test:
    test_df = pd.read_csv(uploaded_test)
    test_df.columns = test_df.columns.str.strip()

    st.write("Test Data Preview")
    st.dataframe(test_df.head())

    predictions = best_model.predict(test_df)
    probabilities = best_model.predict_proba(test_df)[:, 1]

    test_df["Prediction"] = le.inverse_transform(predictions)
    test_df["Probability"] = probabilities

    st.success("Predictions Completed")

    st.dataframe(test_df.head())

    csv = test_df.to_csv(index=False).encode()
    st.download_button(
        "Download Predictions CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )
