# ============================================================
# BITS ML ASSIGNMENT - FINAL OPTIMIZED TRAINING SCRIPT
# Author: Sathya
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv("data.csv")  # Change if needed
print("Dataset shape:", df.shape)

target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# 🔥 Encode target if categorical (Fixes XGBoost issue)
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)


# ============================================================
# 2️⃣ TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# ============================================================
# 3️⃣ PREPROCESSING
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# ============================================================
# 4️⃣ MODELS (OPTIMIZED FOR SIZE)
# ============================================================

models = {

    "logistic": LogisticRegression(max_iter=1000),

    "decision_tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "knn": KNeighborsClassifier(n_neighbors=5),

    "naive_bayes": GaussianNB(),

    "random_forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "xgboost": XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0
    )
}


# ============================================================
# 5️⃣ CREATE MODEL FOLDER
# ============================================================

os.makedirs("model", exist_ok=True)

results = []


# ============================================================
# 6️⃣ TRAINING LOOP
# ============================================================

for name, model in models.items():

    print("\n============================")
    print(f"Training {name.upper()}...")
    print("============================")

    if name == "naive_bayes":
        # GaussianNB requires dense matrix
        X_train_processed = preprocessor.fit_transform(X_train).toarray()
        X_test_processed = preprocessor.transform(X_test).toarray()

        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        # Save preprocessor + model
        joblib.dump(preprocessor, "model/preprocessor.pkl", compress=3)
        joblib.dump(model, f"model/{name}.pkl", compress=3)

        cv_mean = cross_val_score(
            model,
            X_train_processed,
            y_train,
            cv=5,
            scoring="accuracy"
        ).mean()

        y_prob = model.predict_proba(X_test_processed)[:, 1]

    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        joblib.dump(pipeline, f"model/{name}.pkl", compress=3)

        cv_mean = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc,
        cv_mean
    ])

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} CV Accuracy: {cv_mean:.4f}")


# ============================================================
# 7️⃣ LEADERBOARD
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
        "CV Mean Accuracy"
    ]
)

results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\nFinal Model Comparison:")
print(results_df)

results_df.to_csv("model/model_comparison.csv", index=False)

print("\n🎯 All models trained and saved successfully.")
print("Saved models are inside the /model directory.")
