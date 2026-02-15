# 🎓 Income Level Prediction Using Machine Learning

## 1️⃣ Problem Statement

The objective of this project is to predict whether an individual earns:

- **<=50K**
- **>50K**

This is a **binary classification problem** where the target variable `income` has two possible values:

- `0` → <=50K  
- `1` → >50K  

Using demographic and employment-related features such as age, education, occupation, capital gain, and hours per week, we aim to build machine learning models that can accurately classify income levels.

---

## 2️⃣ Dataset Description

- **Source:** UCI Adult Census Income Dataset  
- **Instances:** 48,842 records  
- **Target Variable:** `income`  
- **Features Used:** Full feature set (after preprocessing & encoding)

### 🔹 Original Features Include:

- Age  
- Workclass  
- Education  
- Marital Status  
- Occupation  
- Relationship  
- Race  
- Sex  
- Capital Gain  
- Capital Loss  
- Hours per Week  
- Native Country  

### 🔹 Preprocessing Steps:

- Missing values handled  
- Categorical variables encoded using One-Hot Encoding  
- Features converted to numerical format  
- Feature scaling applied where necessary  

---

## 3️⃣ Models Used & Performance Comparison

We implemented and evaluated 6 classification models using the full feature set.

| ML Model | Accuracy | ROC AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|---------|-----------|--------|----------|------|
| Logistic Regression | 0.8543 | 0.9057 | 0.8479 | 0.8543 | 0.8487 | 0.5769 |
| Decision Tree | 0.8606 | 0.7538 | 0.8191 | 0.8174 | 0.8182 | 0.5030 |
| kNN | 0.8331 | 0.8552 | 0.8281 | 0.8331 | 0.8300 | 0.5267 |
| Naive Bayes | 0.6240 | 0.8267 | 0.8179 | 0.6240 | 0.6481 | 0.3900 |
| Random Forest | 0.8566 | 0.9037 | 0.8538 | 0.8591 | 0.8550 | 0.5953 |
| XGBoost | **0.8741** | **0.9298** | 0.8696 | 0.8741 | 0.8700 | **0.6377** |

---

## 4️⃣ Observations

### 🥇 Best Performing Model: XGBoost
- Highest accuracy: **87.41%**
- Strong ROC AUC: **0.93**
- Best MCC score: **0.64**
- Excellent generalization ability

### 🥈 Random Forest
- Strong performance: **85.66%**
- Effectively handles nonlinear relationships

### 🥉 Logistic Regression
- Competitive baseline: **85.43%**
- Indicates partial linear separability in income prediction

### ⚠ Naive Bayes
- High precision but lower recall
- Simplifying assumptions reduce performance

### 🔍 kNN
- Moderate performance
- Sensitive to scaling and dataset size

---

## 5️⃣ Key Insights

- Ensemble models outperform single models.
- Important predictors include:
  - Education level  
  - Capital gain  
  - Hours per week  
  - Occupation  
- MCC is important due to moderate dataset imbalance.

---

## 6️⃣ Model Interpretation

### Logistic Regression
Learns weighted linear relationships between features and income.

### Decision Tree
Creates rule-based splits using feature thresholds.

### Random Forest
Combines multiple decision trees to reduce variance and improve stability.

### XGBoost
Sequential boosting model that improves errors iteratively, leading to strong performance.

---

## 7️⃣ Conclusion

The Income Classification problem can be effectively solved using ensemble learning methods.

> **XGBoost provides the best balance of accuracy, robustness, and generalization.**

The deployed Streamlit dashboard includes:

- Pre-trained model comparison  
- Live income prediction  
- Test dataset evaluation  
- Detailed metric interpretation  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_models.py   # Optional (if models not pre-trained)
streamlit run app.py
