# ==============================================================================
# Streamlit Web Application: Telco Churn Classification & Model Evaluation
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------------------------
# 1. Page Setup & Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Model Evaluator",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path.cwd()
MODEL_DIR = BASE_DIR / "model"
OPTIMIZED_THRESHOLD = 0.45

# ------------------------------------------------------------------------------
# 2. Artifact Loading Pipeline
# ------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    models = {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
        "Decision Tree": joblib.load(MODEL_DIR / "decision_tree.pkl"),
        "KNN": joblib.load(MODEL_DIR / "knn.pkl"),
        "Naive Bayes": joblib.load(MODEL_DIR / "naive_bayes.pkl"),
        "Support Vector Machine": joblib.load(MODEL_DIR / "support_vector_machine.pkl")
    }
    return scaler, models

try:
    scaler, models = load_pipeline_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts from `{MODEL_DIR}`: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 3. Sidebar Controls
# ------------------------------------------------------------------------------
st.sidebar.header("Upload & Select Model")

# Feature a: Dataset upload option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Feature b: Model selection dropdown
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
model = models[selected_model_name]

# ------------------------------------------------------------------------------
# 4. Main Interface Engine
# ------------------------------------------------------------------------------
st.title("📊 Telco Customer Churn Classification Dashboard")
st.write("Upload test dataset (CSV) to evaluate saved machine learning classifiers.")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Clean whitespace from CSV column names
    test_df.columns = test_df.columns.str.strip()
    
    # Case-insensitive check for target column 'Churn'
    target_col = None
    for col in test_df.columns:
        if col.lower() == 'churn':
            target_col = col
            break
            
    has_target = target_col is not None
    
    if has_target:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].astype(int)
    else:
        X_test = test_df.copy()

    # Preprocessing matching training configuration
    if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        X_test_proc = scaler.transform(X_test)
    else:
        X_test_proc = X_test.values

    # Inference execution (without retraining)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_proc)[:, 1]
    else:
        y_proba = model.predict(X_test_proc)

    y_pred = (y_proba >= OPTIMIZED_THRESHOLD).astype(int)

    # --------------------------------------------------------------------------
    # Scenario A: Target Column Present -> Display Rubric Metrics & Visuals
    # --------------------------------------------------------------------------
    if has_target:
        st.markdown(f"### Evaluation Metrics — `{selected_model_name}`")
        
        # Feature c: Display of evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC Score", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        # Feature d: Confusion matrix & classification report
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["No Churn (0)", "Churn (1)"],
                        yticklabels=["No Churn (0)", "Churn (1)"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)

        with col_right:
            st.subheader("Classification Report")
            report_dict = classification_report(
                y_test, y_pred, 
                target_names=["No Churn (0)", "Churn (1)"], 
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # Scenario B: Target Column Missing -> Output Batch Predictions
    # --------------------------------------------------------------------------
    else:
        st.info("ℹ️ Target column `'Churn'` not found. Outputting batch predictions below:")
        output_df = test_df.copy()
        output_df['Predicted_Probability'] = np.round(y_proba, 4)
        output_df['Predicted_Churn'] = y_pred

        st.dataframe(output_df, use_container_width=True)

        csv_export = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_export,
            file_name=f"churn_predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            type="primary"
        )

else:
    st.info("👈 Please upload `test_data.csv` in the sidebar to run model evaluations.")