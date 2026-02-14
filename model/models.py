from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import numpy as np


def build_pipeline(model, X, force_dense=False):

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_cols
            )
        ]
    )

    steps = [("preprocessor", preprocessor)]

    # Force dense conversion for Naive Bayes
    if force_dense:
        steps.append(
            ("to_dense", FunctionTransformer(lambda x: np.asarray(x)))
        )

    steps.append(("classifier", model))

    return Pipeline(steps)


def get_model(model_name, X):

    if model_name == "Naive Bayes":
        return build_pipeline(
            GaussianNB(),
            X,
            force_dense=True   # IMPORTANT
        )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(n_neighbors=5),

        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric="logloss",
            random_state=42
        )
    }

    return build_pipeline(models[model_name], X)
