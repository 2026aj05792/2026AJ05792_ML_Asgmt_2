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
# 1. Page Configuration & Custom Styling (DARK UPLOAD CARD CONTRAST FIX)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main layout background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headers - Single line title fit */
    h1 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-size: 2.1rem !important;
        font-weight: 700;
        margin-bottom: 0.2rem;
        white-space: nowrap;
    }
    
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    
    /* General sidebar text, headers, and labels */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #f8fafc !important;
    }

    /* Fix File Uploader Dropzone */
    [data-testid="stFileUploader"] section {
        background-color: #1e293b !important;
        border: 1px dashed #475569 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #0369a1 !important;
    }

    /* CRITICAL FIX: Make the uploaded file pill dark with high-contrast text & icons */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #f8fafc !important;
    }

    [data-testid="stFileUploader"] ul, 
    [data-testid="stFileUploader"] li, 
    [data-testid="stFileUploader"] div[role="button"],
    div[data-testid="stFileUploaderFile"] {
        background-color: #334155 !important;
        border-radius: 6px !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] svg {
        fill: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* -------------------------------------------------------------------------
       STRICT SINGLE-LINE (NO-WRAP) KPI CARDS
       ------------------------------------------------------------------------- */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0284c7;
        border-radius: 8px;
        padding: 10px 4px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }

    .kpi-title {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        white-space: nowrap !important;
        word-break: keep-all !important;
        letter-spacing: -0.02em;
    }

    /* Dataframe visual styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.subheader("1. Data Input")
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)", 
        type=["csv"], 
        help="Upload test_data.csv containing customer features."
    )
    
    st.subheader("2. Model Selection")
    selected_model_name = st.selectbox(
        "Choose Classifier",
        list(models.keys()),
        index=4,  # Defaults to Support Vector Machine
        help="Select any of the 5 trained ML classifiers to run evaluations."
    )

model = models[selected_model_name]

# ------------------------------------------------------------------------------
# 4. Main Interface Header & Engine
# ------------------------------------------------------------------------------
st.title("⚡ Telco Churn Intelligence Dashboard")
st.markdown("Evaluate trained classifiers on unseen test data or perform real-time batch customer churn predictions.")
st.markdown("---")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()
    
    # Identify target column 'Churn'
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

    # Column alignment to match feature order expected by scaler/model
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
        missing_cols = set(expected_features) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[expected_features]

    # Preprocessing: Scale features for distance/linear models
    if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        try:
            X_test_proc = scaler.transform(X_test)
        except ValueError:
            X_test_proc = scaler.transform(X_test.values)
    else:
        X_test_proc = X_test.copy()

    # Synchronized Probability Inference Engine
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_proc)[:, 1]
    elif hasattr(model, "decision_function"):
        d_scores = model.decision_function(X_test_proc)
        y_proba = 1 / (1 + np.exp(-d_scores))
    else:
        y_proba = model.predict(X_test_proc)

    # Classification decision thresholding
    y_pred = (y_proba >= OPTIMIZED_THRESHOLD).astype(int)

    # --------------------------------------------------------------------------
    # Scenario A: Target Column Present -> Evaluation View
    # --------------------------------------------------------------------------
    if has_target:
        st.subheader(f"🎯 Performance Evaluation: `{selected_model_name}`")
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # 6 Single-Line Metric Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Accuracy</div>
                <div class="kpi-value">{acc:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        c2.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">AUC Score</div>
                <div class="kpi-value">{auc:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        c3.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Precision</div>
                <div class="kpi-value">{prec:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        c4.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">Recall</div>
                <div class="kpi-value">{rec:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        c5.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">F1 Score</div>
                <div class="kpi-value">{f1:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        c6.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-title">MCC Score</div>
                <div class="kpi-value">{mcc:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("##### 🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"]
            )
            plt.xlabel("Predicted Label", fontsize=10, fontweight='bold', labelpad=8)
            plt.ylabel("True Label", fontsize=10, fontweight='bold', labelpad=8)
            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            st.markdown("##### 📋 Detailed Classification Report")
            report_dict = classification_report(
                y_test, y_pred, 
                target_names=["Retained (0)", "Churned (1)"], 
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # Scenario B: Target Column Missing -> Batch Prediction View
    # --------------------------------------------------------------------------
    else:
        st.info("ℹ️ Target column `'Churn'` not found. Displaying batch predictions:")
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
    # Landing State
    st.info("👈 Upload your `test_data.csv` using the left sidebar to begin model evaluation.")