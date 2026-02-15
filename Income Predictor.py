import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
TARGET = "income"

# Encode target
label_encoder = LabelEncoder()
df[TARGET] = label_encoder.fit_transform(df[TARGET])

# One-hot encode features
X = pd.get_dummies(df.drop(columns=[TARGET]))
y = df[TARGET]
train_columns = X.columns

# ==========================================================
# TRAIN MODELS (FAST + CACHED)
# ==========================================================
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=400),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "mcc": matthews_corrcoef(y_test, preds)
        }

    return results

models_dict = train_models()

# ==========================================================
# SIDEBAR CONFIGURATION
# ==========================================================
st.sidebar.title("⚙ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models_dict.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)"
)

selected_model = models_dict[model_name]["model"]

# ==========================================================
# MAIN HEADER
# ==========================================================
st.title("🎓 BITS ML Classification Dashboard")
st.caption("Pre-trained models • Upload test dataset to evaluate")

# ==========================================================
# TARGET DISTRIBUTION (SMALL CLEAN BAR)
# ==========================================================
st.subheader("🎯 Target Distribution")

dist = df[TARGET].value_counts().reset_index()
dist.columns = ["Class", "Count"]
dist["Class"] = label_encoder.inverse_transform(dist["Class"])

chart = (
    alt.Chart(dist)
    .mark_bar()
    .encode(
        x="Class",
        y="Count",
        text="Count"
    )
    .properties(width=300, height=250)
)

st.altair_chart(chart, use_container_width=False)

# ==========================================================
# MODEL COMPARISON TABLE
# ==========================================================
st.subheader("🏆 Pre-Trained Model Comparison")

table_data = []
for name, values in models_dict.items():
    table_data.append([
        name,
        values["accuracy"],
        values["precision"],
        values["recall"],
        values["f1"],
        values["roc_auc"],
        values["mcc"]
    ])

results_df = pd.DataFrame(
    table_data,
    columns=["Model","Accuracy","Precision","Recall","F1","ROC AUC","MCC"]
).sort_values("Accuracy", ascending=False)

st.dataframe(
    results_df.style
        .format("{:.3f}")
        .background_gradient(cmap="Blues", subset=["Accuracy","F1","ROC AUC"]),
    use_container_width=True
)

# ==========================================================
# LIVE INCOME PREDICTOR
# ==========================================================
st.subheader("🔮 Live Income Predictor")

corr = df.corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
top_features = corr.index[1:6]

cols = st.columns(3)
user_input = {}

for i, feature in enumerate(top_features):
    with cols[i % 3]:
        user_input[feature] = st.slider(
            feature,
            int(df[feature].min()),
            int(df[feature].max()),
            int(df[feature].mean())
        )

if st.button("Predict Income"):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    pred = selected_model.predict(input_df)[0]
    prob = selected_model.predict_proba(input_df)[0][1]
    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Income: {label}")
    st.metric(">50K Probability", f"{prob:.2%}")

# ==========================================================
# TEST DATA EVALUATION
# ==========================================================
if uploaded_file:
    st.subheader("📊 Test Dataset Evaluation")

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    if TARGET not in test_df.columns:
        st.error("Target column missing in test dataset.")
    else:
        if st.button("Apply Model on Test Data"):

            test_df[TARGET] = label_encoder.transform(test_df[TARGET])
            X_test = pd.get_dummies(test_df.drop(columns=[TARGET]))
            X_test = X_test.reindex(columns=train_columns, fill_value=0)
            y_test = test_df[TARGET]

            preds = selected_model.predict(X_test)
            probs = selected_model.predict_proba(X_test)[:,1]

            cm = confusion_matrix(y_test, preds)
            cm_df = pd.DataFrame(cm)

            st.write("Confusion Matrix")
            st.dataframe(cm_df)

            st.write("Metrics")

            c1,c2,c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy_score(y_test,preds):.3f}")
            c2.metric("F1 Score", f"{f1_score(y_test,preds):.3f}")
            c3.metric("ROC AUC", f"{roc_auc_score(y_test,probs):.3f}")

            c4,c5,c6 = st.columns(3)
            c4.metric("Precision", f"{precision_score(y_test,preds):.3f}")
            c5.metric("Recall", f"{recall_score(y_test,preds):.3f}")
            c6.metric("MCC", f"{matthews_corrcoef(y_test,preds):.3f}")

            st.success("Evaluation Complete")
