import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import train_test_split, cross_val_score
from model.models import get_model


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="BITS ML Dashboard",
    layout="wide",
    page_icon="📊"
)

# ------------------ PREMIUM BITS THEME ------------------
st.markdown("""
<style>
body {
    background-color: #0B1F3A;
}

h1, h2, h3 {
    color: #F4A300;
}

.metric-container {
    background-color: #13294B;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #1E3A8A;
}

.metric-label {
    font-size: 14px;
    color: #AFCBFF;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #F4A300;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

st.title("🎓 BITS ML Classification Dashboard")
st.markdown("Train, evaluate and visualize multiple classification models.")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.header("⚙ Model Configuration")

    target_column = st.sidebar.selectbox("Select Target Column", df.columns)

    model_name = st.sidebar.selectbox(
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

    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    run_button = st.sidebar.button("🚀 Apply & Train Model")

    # ------------------ APPLY BUTTON LOGIC ------------------
    if run_button:

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        if y.dtype == "object":
            y = pd.factorize(y)[0]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        with st.spinner("Training model..."):

            model = get_model(model_name, X_train)
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted")
        recall = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except:
            auc = None

        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()

        # ------------------ PREMIUM METRIC CARDS ------------------
        st.markdown("## 📊 Model Performance")

        cols = st.columns(6)
        metrics = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 Score", f1),
            ("MCC", mcc),
            ("CV Accuracy", cv_mean)
        ]

        for col, (label, value) in zip(cols, metrics):
            col.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        if auc:
            st.markdown(f"### 📈 ROC AUC Score: `{auc:.4f}`")

        # ------------------ CLASSIFICATION REPORT ------------------
        st.markdown("## 📋 Classification Report")

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)

        styled = report_df.style \
            .background_gradient(cmap="Blues") \
            .set_properties(**{
                "background-color": "#13294B",
                "color": "white",
                "border-color": "#1E3A8A"
            })

        st.dataframe(styled, use_container_width=True)

        # ------------------ CONFUSION MATRIX ------------------
        st.markdown("## 🧮 Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            annot_kws={"size": 20, "weight": "bold"},
            linewidths=2
        )
        st.pyplot(fig_cm)

        # ------------------ METRIC BAR CHART ------------------
        st.markdown("## 📊 Metric Comparison")

        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "MCC"],
            "Score": [accuracy, precision, recall, f1, mcc]
        })

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        sns.barplot(data=metric_df, x="Metric", y="Score", palette="viridis")
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title("Performance Overview", fontsize=14, weight="bold")
        st.pyplot(fig_bar)

        # ------------------ ROC CURVE ------------------
        if auc:
            st.markdown("## 📈 ROC Curve")

            fpr, tpr, _ = roc_curve(y_test, probs)

            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, linewidth=3, label=f"AUC = {auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # ------------------ FEATURE IMPORTANCE ------------------
        if model_name in ["Random Forest", "XGBoost"]:

            st.markdown("## 🌟 Feature Importance")

            importances = model.named_steps["classifier"].feature_importances_
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()

            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
            sns.barplot(data=fi_df, x="Importance", y="Feature", palette="magma")
            st.pyplot(fig_fi)
