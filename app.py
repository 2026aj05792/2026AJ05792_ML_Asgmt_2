import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("📊 Telco Customer Churn Classification Dashboard")
st.markdown("Upload test dataset (CSV) to evaluate saved machine learning classifiers.")

# Sidebar Configuration
st.sidebar.header("Upload & Select Model")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if 'Churn' not in test_df.columns:
        st.error("Error: Target column 'Churn' is missing from uploaded CSV.")
    else:
        st.subheader("📌 Raw Uploaded Test Data Preview")
        st.dataframe(test_df.head())

        # Split features and target
        y_test = test_df['Churn']
        X_test = test_df.drop(columns=['Churn'])

        scaler = joblib.load('model/scaler.pkl')
        model = joblib.load(model_options[selected_model_name])

        if selected_model_name in ['Logistic Regression', 'KNN', 'Support Vector Machine']:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"📈 Evaluation Metrics: {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn', 'Churn'],
                        yticklabels=['No Churn', 'Churn'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        with col2:
            st.subheader("📋 Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose())
else:
    st.info("👈 Please upload `test_data.csv` in the sidebar to run model evaluations.")