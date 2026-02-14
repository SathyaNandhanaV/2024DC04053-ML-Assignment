# ============================================================
# BITS ML ASSIGNMENT - CLOUD SAFE STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="BITS ML Deployment", layout="wide")

st.title("🎓 BITS ML Assignment - Cloud Deployment")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode target
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ============================================================
# TRAIN MODEL INSIDE STREAMLIT
# ============================================================

@st.cache_resource
def train_best_model():

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0
        )
    }

    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        cv_score = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy").mean()

        if cv_score > best_score:
            best_score = cv_score
            best_model = pipeline
            best_name = name

    best_model.fit(X, y)

    return best_name, best_model, best_score


model_name, model, score = train_best_model()

st.success(f"🔥 Best Model Selected: {model_name}")
st.info(f"Cross Validation Accuracy: {score:.4f}")

# ============================================================
# USER INPUT
# ============================================================

st.sidebar.header("Enter Feature Values")

user_input = {}

for column in X.columns:
    if X[column].dtype == "object":
        user_input[column] = st.sidebar.selectbox(
            column,
            sorted(X[column].unique())
        )
    else:
        user_input[column] = st.sidebar.number_input(
            column,
            value=float(X[column].mean())
        )

input_df = pd.DataFrame([user_input])

# ============================================================
# PREDICTION
# ============================================================

st.subheader("Prediction")

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.success(f"Predicted Class: {prediction}")
    st.info(f"Prediction Probability: {probability:.4f}")
