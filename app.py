# ==============================================================================
# Streamlit Web Application: Telco Churn Classification & Model Evaluation
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
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
# 1. Page Configuration & Custom Styling
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
    
    /* Sidebar text, headers, and labels */
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

    /* Make uploaded file pill dark with high-contrast text & icons */
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
       STRICT SINGLE-LINE (NO-WRAP) KPI CARDS - 6 CORE METRICS
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
        "Decision Tree Classifier": joblib.load(MODEL_DIR / "decision_tree.pkl"),
        "K-Nearest Neighbor Classifier": joblib.load(MODEL_DIR / "knn.pkl"),
        "Naive Bayes Classifier - Gaussian": joblib.load(MODEL_DIR / "naive_bayes.pkl"),
        "Support Vector Machine": joblib.load(MODEL_DIR / "support_vector_machine.pkl")
    }
    return scaler, models

try:
    scaler, models = load_pipeline_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts from `{MODEL_DIR}`: {e}")
    st.stop()

# Helper function to process features & run model inference
def run_model_inference(m_name, m_obj, df_features, scaler_obj):
    if hasattr(scaler_obj, "feature_names_in_"):
        expected_features = list(scaler_obj.feature_names_in_)
        missing_cols = set(expected_features) - set(df_features.columns)
        for c in missing_cols:
            df_features[c] = 0
        df_proc = df_features[expected_features]
    else:
        df_proc = df_features.copy()

    if m_name in ["Logistic Regression", "K-Nearest Neighbor Classifier", "Support Vector Machine"]:
        try:
            X_proc = scaler_obj.transform(df_proc)
        except ValueError:
            X_proc = scaler_obj.transform(df_proc.values)
    else:
        X_proc = df_proc.copy()

    if hasattr(m_obj, "predict_proba"):
        probs = m_obj.predict_proba(X_proc)[:, 1]
    elif hasattr(m_obj, "decision_function"):
        d_scores = m_obj.decision_function(X_proc)
        probs = 1 / (1 + np.exp(-d_scores))
    else:
        probs = m_obj.predict(X_proc)

    preds = (probs >= OPTIMIZED_THRESHOLD).astype(int)
    return preds, probs

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

    # Active selected model predictions
    y_pred, y_proba = run_model_inference(selected_model_name, model, X_test, scaler)

    # --------------------------------------------------------------------------
    # Scenario A: Target Column Present -> Evaluation View
    # --------------------------------------------------------------------------
    if has_target:
        st.subheader(f"🎯 Performance Evaluation: `{selected_model_name}`")
        
        # 6 Mandatory Rubric Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_samples = len(y_test)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        total_churners_caught = tp
        total_actual_churners = tp + fn
        pct_churners_caught = (tp / total_actual_churners * 100) if total_actual_churners > 0 else 0.0

        # Top Grid: Exactly 6 Core Metric KPI Cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.markdown(f'<div class="kpi-card"><div class="kpi-title">Accuracy</div><div class="kpi-value">{acc:.4f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-title">AUC Score</div><div class="kpi-value">{auc:.4f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Precision</div><div class="kpi-value">{prec:.4f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card"><div class="kpi-title">Recall</div><div class="kpi-value">{rec:.4f}</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="kpi-card"><div class="kpi-title">F1 Score</div><div class="kpi-value">{f1:.4f}</div></div>', unsafe_allow_html=True)
        c6.markdown(f'<div class="kpi-card"><div class="kpi-title">MCC Score</div><div class="kpi-value">{mcc:.4f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("##### 🔲 Confusion Matrix (Counts & Total %)")
            
            # Formatted Annotations showing Count + Percentage of Total Dataset
            annot_matrix = np.array([
                [f"{tn}\n({tn/total_samples:.1%})", f"{fp}\n({fp/total_samples:.1%})"],
                [f"{fn}\n({fn/total_samples:.1%})", f"{tp}\n({tp/total_samples:.1%})"]
            ])

            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#f8f9fa')
            
            sns.heatmap(
                cm, annot=annot_matrix, fmt="", cmap="YlGnBu", cbar=False, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"]
            )
            plt.xlabel("Predicted Label", fontsize=10, fontweight='bold', labelpad=8)
            plt.ylabel("True Label", fontsize=10, fontweight='bold', labelpad=8)
            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            st.markdown("##### 📋 Class Breakdown (% Rates)")
            
            st.info(f"🎯 **Churn Detection Rate:** **{pct_churners_caught:.2f}%** ({total_churners_caught} / {total_actual_churners} actual churners caught)")
            st.warning(f"⚠️ **Missed Churners (FNR):** **{fnr*100:.2f}%** | **False Alarm Rate (FPR):** **{fpr*100:.2f}%**")

            # Clean percentage breakdown table
            report_dict = classification_report(
                y_test, y_pred, 
                target_names=["Retained (0)", "Churned (1)"], 
                output_dict=True
            )
            
            pct_table_data = {
                "Class Name": ["Retained (0)", "Churned (1)"],
                "Precision Rate (%)": [f"{report_dict['Retained (0)']['precision']*100:.2f}%", f"{report_dict['Churned (1)']['precision']*100:.2f}%"],
                "Recall Rate (%)": [f"{report_dict['Retained (0)']['recall']*100:.2f}%", f"{report_dict['Churned (1)']['recall']*100:.2f}%"],
                "F1 Score (%)": [f"{report_dict['Retained (0)']['f1-score']*100:.2f}%", f"{report_dict['Churned (1)']['f1-score']*100:.2f}%"],
                "Sample Count": [int(report_dict['Retained (0)']['support']), int(report_dict['Churned (1)']['support'])]
            }
            pct_df = pd.DataFrame(pct_table_data)
            st.dataframe(pct_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # ----------------------------------------------------------------------
        # Model Comparison Bar Chart Section (NATIVE ALTAIR INTERACTIVE CHART)
        # ----------------------------------------------------------------------
        st.subheader("📊 Classifier Comparison across Test Dataset (%)")
        
        comparison_records = []
        for name, m_obj in models.items():
            p_pred, p_proba = run_model_inference(name, m_obj, X_test, scaler)
            comparison_records.append({
                "Model": name,
                "Accuracy (%)": round(accuracy_score(y_test, p_pred) * 100, 2),
                "Precision (%)": round(precision_score(y_test, p_pred, zero_division=0) * 100, 2),
                "Recall (%)": round(recall_score(y_test, p_pred, zero_division=0) * 100, 2),
                "F1 Score (%)": round(f1_score(y_test, p_pred, zero_division=0) * 100, 2),
                "AUC (%)": round(roc_auc_score(y_test, p_proba) * 100, 2)
            })
            
        comp_df = pd.DataFrame(comparison_records)
        comp_melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Percentage (%)")

        chart = alt.Chart(comp_melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Model:N', title=None, axis=alt.Axis(labelAngle=0, labelFontWeight='bold', labelFontSize=11, labelColor='#1e293b')),
            y=alt.Y('Percentage (%):Q', title='Score Percentage (%)', scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('Metric:N', scale=alt.Scale(
                domain=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)', 'AUC (%)'],
                range=['#1D4ED8', '#0284C7', '#059669', '#D97706', '#475569']
            ), legend=alt.Legend(title="Evaluation Metrics", orient="top")),
            xOffset='Metric:N',
            tooltip=['Model', 'Metric', 'Percentage (%)']
        ).properties(
            height=380
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            gridColor='#E2E8F0',
            domainColor='#CBD5E1'
        )

        st.altair_chart(chart, use_container_width=True)

    # --------------------------------------------------------------------------
    # Scenario B: Target Column Missing -> Batch Prediction View
    # --------------------------------------------------------------------------
    else:
        st.info("ℹ️ Target column `'Churn'` not found. Displaying batch predictions:")
        output_df = test_df.copy()
        output_df['Predicted_Probability_%'] = np.round(y_proba * 100, 2)
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