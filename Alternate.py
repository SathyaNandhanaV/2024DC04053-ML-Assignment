import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(layout="wide")

# ------------------------------------------------
# CUSTOM CSS (Compact Professional Look)
# ------------------------------------------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 700;
}
.subtle {
    color: #9aa0a6;
}
.hero-box {
    background: linear-gradient(135deg, #1f2a44, #111827);
    padding: 30px;
    border-radius: 16px;
    text-align: center;
}
.hero-text {
    font-size: 40px;
    font-weight: 800;
    color: #00d4ff;
}
.small-title {
    font-size: 18px;
    font-weight: 600;
}
.metric-card {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown('<div class="big-title">🎓 BITS ML Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Pre-trained models • Upload test dataset to evaluate</div>', unsafe_allow_html=True)
st.markdown("---")


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv("Data.csv")
df.columns = df.columns.str.strip()
target = "income"

X = df.drop(target, axis=1)
y = df[target]

if y.dtype == "object":
    y = y.astype("category").cat.codes

# ------------------------------------------------
# TARGET DISTRIBUTION (COMPACT)
# ------------------------------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("### 🎯 Target Distribution")

    fig, ax = plt.subplots(figsize=(3,3))
    y.value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

with col2:
    st.markdown("### 📊 Class Counts")
    st.dataframe(y.value_counts().to_frame("Count"))


st.markdown("---")


# ------------------------------------------------
# PRETRAINED RESULTS (ASSUME ALREADY COMPUTED)
# ------------------------------------------------
st.markdown("## 🏆 Pre-Trained Model Leaderboard")

leaderboard_df = st.session_state.get("leaderboard_df")

if leaderboard_df is not None:

    best_model = leaderboard_df.iloc[0]

    colA, colB, colC = st.columns(3)

    colA.markdown('<div class="metric-card"><div class="small-title">Best Model</div><div class="hero-text">'+best_model["Model"]+'</div></div>', unsafe_allow_html=True)
    colB.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    colC.metric("ROC AUC", f"{best_model['ROC AUC']:.4f}")

    numeric_cols = leaderboard_df.select_dtypes(include="number").columns

    styled = leaderboard_df.style \
        .format({col: "{:.4f}" for col in numeric_cols}) \
        .background_gradient(subset=numeric_cols, cmap="viridis")

    st.dataframe(styled, use_container_width=True)


st.markdown("---")


# ------------------------------------------------
# UPLOAD TEST DATA
# ------------------------------------------------
st.markdown("## 📂 Evaluate on Test Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    if y_test.dtype == "object":
        y_test = y_test.astype("category").cat.codes

    model = st.session_state.get("selected_model")

    if model:

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1]

        st.markdown("---")

        # HERO RESULT
        st.markdown("## 🔮 Classification Result")

        final_acc = accuracy_score(y_test, preds)

        st.markdown(
            f"""
            <div class="hero-box">
                <div class="small-title">Overall Accuracy</div>
                <div class="hero-text">{final_acc:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # METRICS GRID
        st.markdown("### 📊 Performance Metrics")

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        m1.metric("Precision", f"{precision_score(y_test, preds):.3f}")
        m2.metric("Recall", f"{recall_score(y_test, preds):.3f}")
        m3.metric("F1", f"{f1_score(y_test, preds):.3f}")
        m4.metric("ROC AUC", f"{roc_auc_score(y_test, probs):.3f}")
        m5.metric("MCC", f"{matthews_corrcoef(y_test, preds):.3f}")
        m6.metric("Samples", len(y_test))

        st.markdown("---")

        # SMALL SIDE-BY-SIDE GRAPHS
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(3,2.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

        with g2:
            st.markdown("##### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, probs)
            fig_roc, ax_roc = plt.subplots(figsize=(3,2.5))
            ax_roc.plot(fpr, tpr)
            ax_roc.plot([0,1],[0,1], linestyle="--")
            st.pyplot(fig_roc)
