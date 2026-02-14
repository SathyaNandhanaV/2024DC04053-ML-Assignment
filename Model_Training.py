import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Data.csv")
df.columns = df.columns.str.strip()
df.replace("?", np.nan, inplace=True)

X = df.drop("income", axis=1)
y = df["income"].map({"<=50K": 0, ">50K": 1})

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns


# =========================
# PREPROCESSORS
# =========================

# Sparse encoder (for most models)
preprocessor_sparse = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))  # sparse_output=True by default
    ]), categorical_cols)
])

# Dense encoder (ONLY for Naive Bayes)
preprocessor_dense = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols)
])


# =========================
# OPTIMIZED MODELS
# =========================
models = {

    "logistic": LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ),

    "decision_tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=5
    ),

    "naive_bayes": GaussianNB(),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs("models", exist_ok=True)

results = []


# =========================
# TRAINING LOOP
# =========================
for name, model in models.items():

    if name == "naive_bayes":
        preprocessor = preprocessor_dense
    else:
        preprocessor = preprocessor_sparse

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    }

    results.append(metrics)

    # Save compressed model (smaller file size)
    joblib.dump(pipeline, f"models/{name}.pkl", compress=3)


# =========================
# PRINT RESULTS
# =========================
results_df = pd.DataFrame(results)

print("\n==============================")
print("MODEL EVALUATION RESULTS")
print("==============================\n")
print(results_df)
