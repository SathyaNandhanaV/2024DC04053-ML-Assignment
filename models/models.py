from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def build_pipeline(model, X):

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False  # IMPORTANT FIX FOR XGBOOST
                ),
                categorical_cols
            )
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline


def get_model(model_name, X):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=50,      # Reduced for speed
            max_depth=10,
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=50,      # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric="logloss",
            random_state=42
        )
    }

    return build_pipeline(models[model_name], X)
