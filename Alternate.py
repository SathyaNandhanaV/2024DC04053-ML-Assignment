import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="BITS ML Dashboard", layout="wide")

st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Pre-trained models • Upload test dataset to evaluate")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    df.replace("?", np.nan, inplace=True)
    return df


df = load_data()

target_column = "income"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert target to numeric if needed
if y.dtype == "object":
    y = y.astype("category").cat.codes


# ------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)


# ------------------------------------------------
# TRAIN MODELS (FAST CLOUD VERSION)
# ------------------------------------------------
@st.cache_resource
def train_all_models():

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=8),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, max_depth=10, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    # Sample for speed (important for cloud)
    X_sample = X.sample(frac=0.7, random_state=42)
    y_sample = y.loc[X_sample.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )

    trained_models = {}
    results = []

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", model)]
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        try:
            probs = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0

        trained_models[name] = pipeline

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy, 4),
                "F1 Score": round(f1, 4),
                "MCC": round(mcc, 4),
                "ROC AUC": round(auc, 4),
            }
        )

        progress.progress((i + 1) / len(models))

    progress.empty()

    leaderboard = pd.DataFrame(results).sort_values(
        by="Accuracy", ascending=False
    )

    return trained_models, leaderboard


models_dict, leaderboard_df = train_all_models()

# ------------------------------------------------
# LEADERBOARD
# ------------------------------------------------
st.header("🏆 Model Leaderboard (Pre-Trained)")

st.dataframe(
    leaderboard_df,
    use_container_width=True,
)

# ------------------------------------------------
# TEST DATA UPLOAD
# ------------------------------------------------
st.header("📂 Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    st.subheader("Preview")
    st.dataframe(test_df.head(), use_container_width=True)

    if target_column not in test_df.columns:
        st.error(f"Target column '{target_column}' not found in uploaded file.")
        st.stop()

    X_test_user = test_df.drop(target_column, axis=1)
    y_test_user = test_df[target_column]

    if y_test_user.dtype == "object":
        y_test_user = y_test_user.astype("category").cat.codes

    selected_model = st.selectbox(
        "Select Model to Evaluate",
        list(models_dict.keys()),
    )

    if st.button("Evaluate Model"):

        model = models_dict[selected_model]

        preds = model.predict(X_test_user)

        accuracy = accuracy_score(y_test_user, preds)
        f1 = f1_score(y_test_user, preds, average="weighted")
        mcc = matthews_corrcoef(y_test_user, preds)

        try:
            probs = model.predict_proba(X_test_user)[:, 1]
            auc = roc_auc_score(y_test_user, probs)
        except:
            auc = 0

        st.header("📊 Test Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")
        col4.metric("ROC AUC", f"{auc:.4f}")

        st.subheader("Classification Report")
        report = classification_report(
            y_test_user, preds, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_user, preds)
        st.dataframe(pd.DataFrame(cm))
