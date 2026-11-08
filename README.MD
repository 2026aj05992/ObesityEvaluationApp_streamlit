# Machine Learning Classification: Obesity Levels

## a. Problem statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict individuals' obesity levels based on their eating habits and physical condition. The goal is to deploy an interactive Streamlit web application that allows users to test pre-trained models on hold-out test data without retraining, completing an end-to-end ML deployment workflow.

## b. Dataset description

The dataset chosen is the **"Estimation of Obesity Levels based on Eating Habits and Physical Condition"** (ID: 544) from the UCI Machine Learning Repository.

- **Features:** 16 features (a mix of numeric and categorical variables such as age, weight, family history of overweight, and caloric consumption).
- **Instances:** 2,111 instances.
- **Target Variable:** `NObeyesdad`, which is a multi-class label categorized into 7 different obesity levels (Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III).

## c. Github Repository Link

https://github.com/2026aj05992/ObesityEvaluationApp_streamlit

## d. Models used:

Below is the Comparison Table with the evaluation metrics calculated for all the 5 models evaluated on the test dataset (20% hold-out).

| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| -------------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| **Logistic Regression**    | 0.8747   | 0.9837 | 0.8729    | 0.8747 | 0.8731 | 0.8539 |
| **Decision Tree**          | 0.9149   | 0.9504 | 0.9166    | 0.9149 | 0.9153 | 0.9007 |
| **KNN**                    | 0.8251   | 0.9661 | 0.8248    | 0.8251 | 0.8170 | 0.7981 |
| **Naive Bayes**            | 0.5083   | 0.8382 | 0.5275    | 0.5083 | 0.4607 | 0.4471 |
| **Support Vector Machine** | 0.9243   | 0.9931 | 0.9264    | 0.9243 | 0.9248 | 0.9118 |

### Observations

---

| ML Model Name                              | Observation about model performance                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **Logistic Regression**                    | Performed surprisingly well as a baseline with ~87.5% accuracy and a very                        |
|                                            | high AUC (0.98), showing that the classes have a good degree of linear                           |
|                                            | separability.                                                                                    |
| ----------------------------               | ------------------------------------------------------------------------------------------------ |
| **Decision Tree**                          | Yielded excellent results (~91.5% accuracy) by effectively capturing non-linear                  |
|                                            | relationships and hierarchical decision boundaries in dietary habits, though its                 |
|                                            | AUC was slightly lower than SVM.                                                                 |
| ---------------------------------------    | -------------------------------------------------------------------------------------            |
| **KNN**                                    | Showed moderate performance (~82.5% accuracy). While scaling helped, KNN struggled               |
|                                            | slightly in the higher dimensional space compared to the tree and SVM models.                    |
| ---------------------------------------    | -------------------------------------------------------------------------------------            |
| **Naive Bayes**                            | Performed the poorest by far (~50.8% accuracy). The strong assumption of feature                 |
|                                            | independence severely hurt the model, as dietary habits and physical traits are                  |
|                                            | highly correlated.                                                                               |
| ---------------------------------------    | -------------------------------------------------------------------------------------            |
| **Support Vector Machine**                 | Provided the most robust performance across all metrics, achieving the highest                   |
|                                            | accuracy (92.43%) and near-perfect class separation (AUC of 0.9931).                             |
| ---------------------------------------    | -------------------------------------------------------------------------------------            |
| **Overall, Winner for your dataset?**      | **Support Vector Machine (SVM)** is the clear winner for this dataset. It achieved               |
|                                            | the highest F1 Score (0.9248), Accuracy, and AUC by effectively handling the multi-              |
|                                            | dimensional feature space and maximizing the margin between the 7 different obesity              |
|                                            | classes.                                                                                         |
| ******************\_\_\_****************** | ******************************************\_******************************************           |
