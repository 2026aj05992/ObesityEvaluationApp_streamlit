import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)

# Set Page title
st.set_page_config(page_title="Obesity Classifier Evaluation", layout="wide", page_icon="📊")

# Custom CSS for better card styling
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Machine Learning Model Evaluation Dashboard")
st.markdown("Evaluate pre-trained classification models on the **Obesity Levels** dataset. Upload your hold-out test data to begin.")

# Define model paths globally
MODEL_OPTIONS = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/support_vector_machine.joblib'
}

# Helper function for modular evaluation
def evaluate_pipeline(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }, y_pred

# Sidebar for inputs
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload test_data.csv", type="csv")
    
    st.markdown("---")
    st.info("**Note:** Ensure the uploaded CSV contains the exact features used during training, plus the 'NObeyesdad' target column.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'NObeyesdad' not in df.columns:
        st.error("Target column 'NObeyesdad' not found. Please upload the correct test_data.csv.")
    else:
        # Prepare data
        X_test = df.drop('NObeyesdad', axis=1)
        y_true_labels = df['NObeyesdad']
        
        # Load the LabelEncoder
        try:
            le = joblib.load('model/label_encoder.joblib')
            y_test = le.transform(y_true_labels)
        except Exception as e:
            st.error(f"Error loading label encoder: {e}. Ensure 'model/label_encoder.joblib' exists in the repository.")
            st.stop()

        # UI Layout: Tabs for cleaner organization
        tab1, tab2, tab3 = st.tabs(["🔍 Single Model View", "📈 Compare All Models", "📄 Dataset Preview"])
        
        with tab1:
            st.subheader("Individual Model Analysis")
            selected_model_name = st.selectbox("Select ML Model to Evaluate", list(MODEL_OPTIONS.keys()))
            
            if st.button("Evaluate Selected Model", type="primary"):
                try:
                    pipeline = joblib.load(MODEL_OPTIONS[selected_model_name])
                    metrics, y_pred = evaluate_pipeline(pipeline, X_test, y_test)
                    
                    # Display Metrics in a clean grid
                    st.markdown("### Performance Metrics")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    col2.metric("AUC", f"{metrics['AUC']:.4f}")
                    col3.metric("Precision", f"{metrics['Precision']:.4f}")
                    col4.metric("Recall", f"{metrics['Recall']:.4f}")
                    col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    col6.metric("MCC", f"{metrics['MCC']:.4f}")
                    
                    st.divider()
                    
                    # Visualizations side-by-side
                    col_cm, col_cr = st.columns([1.2, 1])
                    
                    with col_cm:
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=le.classes_, yticklabels=le.classes_, cbar=False)
                        plt.ylabel('Actual Class')
                        plt.xlabel('Predicted Class')
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                        
                    with col_cr:
                        st.markdown("### Classification Report")
                        report = classification_report(y_test, y_pred, target_names=le.classes_)
                        st.text(report)
                        
                except Exception as e:
                    st.error(f"Error loading or evaluating model: {e}")

        with tab2:
            st.subheader("Model Comparison Dashboard")
            st.write("Compare the evaluation metrics of all 5 models side-by-side.")
            
            if st.button("Generate Comparison Graph", type="primary"):
                with st.spinner("Evaluating all models..."):
                    all_results = []
                    
                    for name, path in MODEL_OPTIONS.items():
                        try:
                            pipeline = joblib.load(path)
                            metrics, _ = evaluate_pipeline(pipeline, X_test, y_test)
                            metrics['Model'] = name
                            all_results.append(metrics)
                        except Exception as e:
                            st.warning(f"Could not evaluate {name}: {e}")
                    
                    if all_results:
                        results_df = pd.DataFrame(all_results)
                        
                        # Display raw table
                        st.dataframe(results_df.set_index('Model').style.highlight_max(axis=0, color='lightgreen'))
                        
                        # Prepare data for seaborn plotting
                        melted_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
                        
                        st.markdown("### Performance Comparison Graph")
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=melted_df, x='Metric', y='Score', hue='Model', palette='viridis', ax=ax2)
                        
                        # Move legend outside the plot
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.title("Evaluation Metrics Across All Classifiers", pad=20)
                        plt.ylim(0, 1.05) # Metrics are mostly 0 to 1
                        plt.tight_layout()
                        
                        st.pyplot(fig2)

        with tab3:
            st.subheader("Test Dataset Preview")
            st.dataframe(df.head(15), use_container_width=True)
            st.write(f"Total instances in test set: **{len(df)}**")

else:
    st.info("👈 Please upload the 'test_data.csv' file using the sidebar to begin evaluation.")