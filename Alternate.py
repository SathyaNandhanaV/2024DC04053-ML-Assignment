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
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="BITS ML Predictor", layout="wide")

st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Preloaded training data. Upload only test data to evaluate.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    df.replace("?", np.nan, inplace=True)
    return df

df = load_data()

target_column = "income"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode target
y = pd.factorize(y)[0]

# ---------------- PREPROCESSOR ----------------
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numerical_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

# ---------------- TRAIN MODELS ----------------
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
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "pipeline": pipeline,
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
            "X_test": X_test,
            "y_test": y_test
        }

    return results


models = train_models()

# ---------------- LEADERBOARD ----------------
st.header("🏆 Model Leaderboard")

leaderboard = pd.DataFrame([
    {
        "Model": name,
        "Accuracy": data["accuracy"],
        "F1 Score": data["f1"],
        "ROC AUC": data["auc"]
    }
    for name, data in models.items()
]).sort_values(by="Accuracy", ascending=False)

st.dataframe(leaderboard, use_container_width=True)

best_model_name = leaderboard.iloc[0]["Model"]
best_pipeline = models[best_model_name]["pipeline"]

st.success(f"🥇 Best Model: {best_model_name}")

# ---------------- BASELINE METRICS ----------------
st.header("📊 Baseline Performance")

y_test = models[best_model_name]["y_test"]
X_test = models[best_model_name]["X_test"]

preds = best_pipeline.predict(X_test)
probs = best_pipeline.predict_proba(X_test)[:, 1]

st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
st.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")
st.metric("ROC AUC", f"{roc_auc_score(y_test, probs):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

# ---------------- TEST DATA UPLOAD ----------------
st.header("📤 Upload Test Dataset")

uploaded_test = st.file_uploader("Upload Test CSV", type="csv")

if uploaded_test:

    test_df = pd.read_csv(uploaded_test)
    test_df.columns = test_df.columns.str.strip()
    test_df.replace("?", np.nan, inplace=True)

    preds = best_pipeline.predict(test_df)
    probs = best_pipeline.predict_proba(test_df)[:, 1]

    test_df["Prediction"] = preds
    test_df["Probability"] = probs

    st.success("Predictions Generated")

    st.dataframe(test_df.head())

    st.download_button(
        "Download Predictions",
        test_df.to_csv(index=False).encode(),
        "predictions.csv",
        "text/csv"
    )
