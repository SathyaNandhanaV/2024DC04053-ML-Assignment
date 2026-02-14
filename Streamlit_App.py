import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

from model.models import get_model

# Optional import for xgboost safety
try:
    from xgboost import XGBClassifier
except:
    pass

st.set_page_config(page_title="Bank ML Classifier", layout="wide")

st.title("Bank Marketing Classification App")

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Encode target if categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        model_name = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]
        )

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        st.write("Training model...")

        model = get_model(model_name, X_train)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # ---- RESULTS ----
        st.subheader("Cross Validation Accuracy")

        cv_scores = cross_val_score(
            get_model(model_name, X_train),
            X,
            y,
            cv=5
        )

        st.write("Mean CV Accuracy:", round(cv_scores.mean(), 4))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, preds))
