import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Bank ML Classifier", layout="wide")

st.title("📊 ML-Assignment-2")
st.markdown("Upload a test dataset and evaluate trained ML models.")

# -----------------------
# Model Selection
# -----------------------

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

model_option = st.selectbox("Select Model", list(MODEL_FILES.keys()))

uploaded_file = st.file_uploader("Upload Test CSV File", type="csv")

# -----------------------
# Load Model Function
# -----------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# -----------------------
# Main Logic
# -----------------------

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.success("File uploaded successfully!")

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        target_column = st.selectbox("Select Target Column", df.columns)

        if target_column:

            X = df.drop(target_column, axis=1)
            y = df[target_column]

            model_path = os.path.join("models", MODEL_FILES[model_option])

            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                st.stop()

            model = load_model(model_path)

            preds = model.predict(X)

            st.subheader("📈 Classification Report")
            st.text(classification_report(y, preds))

            st.subheader("🧮 Confusion Matrix")
            st.write(confusion_matrix(y, preds))

    except Exception as e:
        st.error(f"Error processing file: {e}")
