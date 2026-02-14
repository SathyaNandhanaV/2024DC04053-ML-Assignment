# ============================================================
# BITS ML ASSIGNMENT - FINAL STREAMLIT APP
# Author: Sathya
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="BITS ML Model Deployment", layout="wide")

st.title("🎓 BITS ML Assignment - Model Deployment")
st.markdown("### 🔥 Automatic Best Model Selection")

# ============================================================
# LOAD BEST MODEL
# ============================================================

@st.cache_resource
def load_best_model():
    leaderboard = pd.read_csv("model/model_comparison.csv")

    best_model_name = leaderboard.iloc[0]["Model"]

    st.success(f"Best Model Automatically Selected: {best_model_name.upper()}")

    if best_model_name == "naive_bayes":
        model = joblib.load(f"model/{best_model_name}.pkl")
        preprocessor = joblib.load("model/preprocessor.pkl")
        return best_model_name, model, preprocessor
    else:
        model = joblib.load(f"model/{best_model_name}.pkl")
        return best_model_name, model, None


model_name, model, preprocessor = load_best_model()

# ============================================================
# LOAD DATA FOR INPUT STRUCTURE
# ============================================================

df = pd.read_csv("data.csv")
target_column = df.columns[-1]
X = df.drop(columns=[target_column])

st.sidebar.header("Enter Input Features")

user_input = {}

for column in X.columns:
    if X[column].dtype == "object":
        user_input[column] = st.sidebar.selectbox(
            column,
            options=sorted(X[column].unique())
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

    if model_name == "naive_bayes":
        processed_input = preprocessor.transform(input_df).toarray()
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
    else:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.success(f"Predicted Class: {prediction}")
    st.info(f"Prediction Probability: {probability:.4f}")

# ============================================================
# LEADERBOARD DISPLAY
# ============================================================

st.subheader("📊 Model Leaderboard")

leaderboard = pd.read_csv("model/model_comparison.csv")
st.dataframe(leaderboard, use_container_width=True)
