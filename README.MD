# Machine Learning Classification: Obesity Levels

## a. Problem statement
The objective of this project is to develop and evaluate multiple machine learning classification models to predict individuals' obesity levels based on their eating habits and physical condition. The goal encompasses the full end-to-end ML workflow: from data preprocessing and modeling to deploying an interactive web application that allows users to test pre-trained models on holdout test data without retraining.

## b. Dataset description
The dataset chosen is the "Estimation of Obesity Levels based on Eating Habits and Physical Condition" sourced from the UCI Machine Learning Repository (ID: 544). It contains 16 features—a mix of numeric and categorical variables such as age, weight, family history, and caloric consumption—and 2,111 instances. The target variable, `NObeyesdad`, is a multi-class label categorized into 7 distinct obesity levels (e.g., Insufficient Weight, Normal Weight, Overweight Level I & II, and Obesity Type I, II, & III).

## c. Github Repository Link
https://github.com/2026aj05992/ObesityEvaluationApp_streamlit

## d. Models used:
*Below are the evaluation metrics for all 5 models based on the hold-out test dataset.*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8747 | 0.9837 | 0.8729 | 0.8747 | 0.8731 | 0.8539 |
| **Decision Tree** | 0.9149 | 0.9504 | 0.9166 | 0.9149 | 0.9153 | 0.9007 |
| **KNN** | 0.8251 | 0.9661 | 0.8248 | 0.8251 | 0.8170 | 0.7981 |
| **Naive Bayes** | 0.5083 | 0.8382 | 0.5275 | 0.5083 | 0.4607 | 0.4471 |
| **Support Vector Machine** | 0.9243 | 0.9931 | 0.9264 | 0.9243 | 0.9248 | 0.9118 |

### Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Served as a solid linear baseline. It captured overarching trends well but struggled slightly with the non-linear boundaries separating the nuanced weight classes. |
| **Decision Tree** | Showed strong ability to interpret conditional dietary habits. While highly accurate on training data, it demonstrated slight overfitting when generalized to the test set. |
| **KNN** | Performance was heavily dependent on the StandardScaler in the preprocessing pipeline. It achieved moderate to high accuracy by grouping similar physical condition profiles. |
| **Naive Bayes** | Demonstrated lower comparative accuracy due to the strict assumption of feature independence, which often does not hold true for correlated dietary and physical habits. |
| **Support Vector Machine** | Provided robust performance and a high AUC score by effectively establishing margins between the 7 classes in a high-dimensional feature space. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** achieved the highest F1 Score (0.9248) and AUC (0.9931). Its ability to balance precision and recall across all 7 multi-class labels made it the most reliable predictor for this specific dataset. |