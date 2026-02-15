# 2024DC04053-ML-Assignment

Income Level Prediction Using Machine Learning
1. Problem Statement
The objective of this project is to predict whether an individual earns <=50K or >50K based on demographic and employment-related attributes. This is a binary classification problem where the target variable 'income' has two possible values: 0 (<=50K) and 1 (>50K). Using structured census data features such as age, education, occupation, capital gain, and hours per week, we aim to build machine learning models that can accurately classify income levels.
2. Dataset Description
Source: UCI Adult Census Income Dataset
Instances: 48,842 records
Features Used: Full feature set after preprocessing
Original Features Include:
•	Age
•	Workclass
•	Education
•	Marital Status
•	Occupation
•	Relationship
•	Race
•	Sex
•	Capital Gain
•	Capital Loss
•	Hours per Week
•	Native Country
Preprocessing Steps: Missing values handled, categorical variables encoded using One-Hot Encoding, features converted to numerical format.
3. Models Used & Performance Comparison
3. Models Used & Performance Comparison
Model	Accuracy	ROC AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8543	0.9057	0.8479	0.8543	0.8487	0.5769
Decision Tree	0.8606	0.7538	0.8191	0.8174	0.8182	0.5030
kNN	0.8331	0.8552	0.8281	0.8331	0.8300	0.5267
Naive Bayes	0.6240	0.8267	0.8179	0.6240	0.6481	0.3900
Random Forest	0.8566	0.9037	0.8538	0.8591	0.8550	0.5953
XGBoost	0.8741	0.9298	0.8696	0.8741	0.8700	0.6377

4. Observations
XGBoost achieved the highest accuracy (87.41%) and best ROC AUC (0.93), demonstrating strong class separability and generalization.
Random Forest performed competitively with strong ensemble stability.
Logistic Regression performed well, indicating partial linear separability in income prediction.
Naive Bayes showed high precision but low recall due to independence assumptions.
kNN showed moderate performance and sensitivity to scaling and dataset size.
5. Key Insights
Ensemble models outperform single models in this dataset.
Important predictors include Education Level, Capital Gain, Hours per Week, and Occupation.
Dataset imbalance makes MCC an important evaluation metric.
6. Conclusion
The Income Classification problem can be effectively solved using ensemble learning techniques. Among all models evaluated, XGBoost provides the best balance of accuracy, robustness, and generalization. The deployed Streamlit dashboard allows pre-trained model comparison, live prediction, test dataset evaluation, and detailed metric interpretation.
