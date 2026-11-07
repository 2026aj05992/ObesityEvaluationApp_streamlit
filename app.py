import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Obesity Level Classification App")
st.write("Upload your test dataset to evaluate the pre-trained Machine Learning models.")

# a. Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    
    if 'NObeyesdad' in df.columns:
        X_test = df.drop('NObeyesdad', axis=1)
        y_true_labels = df['NObeyesdad']
        
        # Load the saved LabelEncoder to transform string labels to integers
        le = joblib.load('model/label_encoder.joblib')
        y_test = le.transform(y_true_labels)
        
        # b. Model selection dropdown
        model_options = {
            'Logistic Regression': 'model/logistic_regression.joblib',
            'Decision Tree': 'model/decision_tree.joblib',
            'KNN': 'model/knn.joblib',
            'Naive Bayes': 'model/naive_bayes.joblib',
            'Support Vector Machine': 'model/support_vector_machine.joblib'
        }
        
        selected_model_name = st.selectbox("Select ML Model to Evaluate", list(model_options.keys()))
        
        if st.button("Evaluate Model"):
            # Load the saved model pipeline
            model_path = model_options[selected_model_name]
            pipeline = joblib.load(model_path)
            
            # Predict using the uploaded test data
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # c. Display of evaluation metrics
            st.subheader(f"Evaluation Metrics: {selected_model_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
            
            # d. Classification report and Confusion matrix
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=le.classes_)
            st.text(report)
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            
    else:
        st.error("Target column 'NObeyesdad' not found. Please upload the correct test_data.csv.")