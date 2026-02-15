import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def safe_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    raise ValueError("Could not read Data.csv")


def get_all_models():   # ✅ NO ARGUMENTS

    df = safe_read_csv("Data.csv")
    df.columns = df.columns.str.strip()

    TARGET = "income"

    label_encoder = LabelEncoder()
    df[TARGET] = label_encoder.fit_transform(df[TARGET])

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
    X = X.astype(float)   # ensure dense
    y = df[TARGET]

    train_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=800
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=42
        )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, train_columns, label_encoder
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def safe_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    raise ValueError("Could not read Data.csv")


def get_all_models():   # ✅ NO ARGUMENTS

    df = safe_read_csv("Data.csv")
    df.columns = df.columns.str.strip()

    TARGET = "income"

    label_encoder = LabelEncoder()
    df[TARGET] = label_encoder.fit_transform(df[TARGET])

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True)
    X = X.astype(float)   # ensure dense
    y = df[TARGET]

    train_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=800
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0,
            random_state=42
        )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, train_columns, label_encoder
