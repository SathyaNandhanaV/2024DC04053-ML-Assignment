# models.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==========================================================
# PREPROCESSOR
# ==========================================================
def build_preprocessor(X):

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


# ==========================================================
# MODEL DEFINITIONS
# ==========================================================
def build_model(model_name):

    if model_name == "Logistic Regression":
        return LogisticRegression(
            solver="liblinear",
            max_iter=800
        )

    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6)

    elif model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5,
            algorithm="ball_tree"
        )

    elif model_name == "Naive Bayes":
        return GaussianNB()

    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )

    elif model_name == "XGBoost":
        return XGBClassifier(
            eval_metric="logloss",
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=42
        )

    else:
        raise ValueError("Invalid model name")


# ==========================================================
# CREATE PIPELINE
# ==========================================================
def get_model(model_name, X):

    preprocessor = build_preprocessor(X)
    model = build_model(model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline


# ==========================================================
# RETURN ALL MODELS
# ==========================================================
def get_all_models(X):

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    models = {}

    for name in model_names:
        models[name] = get_model(name, X)

    return models
