import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

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

label_encoder = LabelEncoder()
df[TARGET] = label_encoder.fit_transform(df[TARGET])

X = pd.get_dummies(df.drop(columns=[TARGET]))
y = df[TARGET]
train_columns = X.columns

# ==========================================================
# TRAIN MODELS
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
# SIDEBAR
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
# HEADER
# ==========================================================
st.title("🎓 Income Predictor")
st.caption("Pre-trained models • Live predictor • Upload test dataset")

# ==========================================================
# LIVE PREDICTOR
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

    c1, c2 = st.columns([2,1])
    c1.success(f"Predicted Income: {label}")
    c2.metric(">50K Probability", f"{prob:.2%}")

# ==========================================================
# MODEL TABLE
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

numeric_cols = ["Accuracy","Precision","Recall","F1","ROC AUC","MCC"]

styled_table = (
    results_df.style
        .format({col: "{:.3f}" for col in numeric_cols})
        .background_gradient(cmap="Blues", subset=numeric_cols)
)

st.dataframe(styled_table, use_container_width=True)

# ==========================================================
# TEST DATA
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

            # 1️⃣ METRICS FIRST
            st.subheader("📊 Test Metrics")

            acc = accuracy_score(y_test,preds)
            f1 = f1_score(y_test,preds)
            roc_auc = roc_auc_score(y_test,probs)
            prec = precision_score(y_test,preds)
            rec = recall_score(y_test,preds)
            mcc = matthews_corrcoef(y_test,preds)

            m1,m2,m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("F1 Score", f"{f1:.3f}")
            m3.metric("ROC AUC", f"{roc_auc:.3f}")

            m4,m5,m6 = st.columns(3)
            m4.metric("Precision", f"{prec:.3f}")
            m5.metric("Recall", f"{rec:.3f}")
            m6.metric("MCC", f"{mcc:.3f}")

            # 2️⃣ PLOTS SIDE BY SIDE
            colA, colB = st.columns(2)

            with colA:
                cm = confusion_matrix(y_test, preds)
                fig1, ax1 = plt.subplots(figsize=(3,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
                ax1.set_title("Confusion Matrix")
                st.pyplot(fig1)

            with colB:
                fpr, tpr, _ = roc_curve(y_test, probs)
                fig2, ax2 = plt.subplots(figsize=(3,3))
                ax2.plot(fpr, tpr)
                ax2.plot([0,1],[0,1],'--')
                ax2.set_title("ROC Curve")
                st.pyplot(fig2)

            # 3️⃣ DATA-BASED SUMMARY
            st.divider()
            st.subheader("📌 Dataset Performance Summary")

            st.write(f"""
            • Model evaluated on **{len(y_test)} test samples**  
            • Accuracy of **{acc:.2%}** indicates overall prediction correctness  
            • ROC AUC of **{roc_auc:.2f}** shows the model's ability to distinguish income classes  
            • Precision ({prec:.2f}) reflects how many predicted high-income cases were correct  
            • Recall ({rec:.2f}) measures how well high-income individuals were identified  
            • MCC score of **{mcc:.2f}** indicates balanced performance across both classes  

            Overall, the model demonstrates {"strong" if acc > 0.85 else "moderate" if acc > 0.75 else "limited"} generalization on this dataset.
            """)
