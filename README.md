# Telco Customer Churn Prediction & Model Evaluation Dashboard

## a. Problem Statement

Customer churn poses a significant challenge for telecommunication companies, directly impacting revenue and long-term business growth. The primary objective of this project is to develop, evaluate, and compare multiple Machine Learning classification models to accurately predict whether a customer will churn (leave the service) or remain retained based on their demographics, subscribed services, account details, and financial attributes. By establishing an automated batch prediction and evaluation dashboard, business teams can proactively identify high-risk churners and deploy targeted retention strategies.

## b. Dataset Description

- **Source & Provenance:** The dataset is sourced from the standard **IBM Sample Data Sets** (commonly hosted and benchmarked on **Kaggle** under *Telco Customer Churn*).
- **Original Dataset Size:** The full dataset contains **7,043 customer records** with **21 feature columns**.
- **Evaluation Test Subset (`test_data.csv`):** A held-out evaluation test set of **1,409 customer records** (80/20 split of the original dataset) was used for model validation.
  * **Target Variable:** `Churn` (`0` = Retained, `1` = Churned).
  * **Class Breakdown:** 1,035 Retained customers (73.46%) and 374 Churned customers (26.54%).

- **Feature Categories:**
  * **Demographics:** Gender, SeniorCitizen, Partner, Dependents.
  * **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
  * **Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.

- **Preprocessing & Feature Alignment:** Categorical variables are numerical/one-hot encoded, and continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled using a standard scaler (`scaler.pkl`). Model inference uses a fixed probability threshold of **0.45**.

## c. Github Repository Link

[https://github.com/2026aj05792/2026AJ05792_ML_Asgmt_2](https://github.com/2026aj05792/2026AJ05792_ML_Asgmt_2)

## d. Models Used & Evaluation Comparison

### 1. Model Performance Comparison Table

The 5 classic classifiers were evaluated on the exact same test dataset (`1,409` records) with a fixed decision threshold of `0.45`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | 0.7239 | 0.8486 | 0.4882 | 0.8316 | 0.6152 | 0.4583 | 
| **Decision Tree Classifier** | 0.6941 | 0.8271 | 0.4583 | 0.8369 | 0.5922 | 0.4236 | 
| **K-Nearest Neighbor Classifier** | 0.7672 | 0.7909 | 0.5665 | 0.5241 | 0.5444 | 0.3890 | 
| **Naive Bayes Classifier - Gaussian** | 0.6537 | 0.8202 | 0.4254 | 0.8690 | 0.5712 | 0.3943 | 
| **Support Vector Machine** | 0.7715 | 0.8262 | 0.5580 | 0.6684 | 0.6083 | 0.4524 | 

### 2. Performance Observations per Model

| ML Model Name | Observation about model performance | 
| ----- | ----- | 
| **Logistic Regression** | Demonstrates the highest overall capability in probability calibration and ranking, achieving the top AUC Score (0.8486), highest F1 Score (0.6152), and highest MCC (0.4583). By incorporating class balancing at the 0.45 threshold, it effectively captures 83.16% of churners while maintaining strong predictive balance. | 
| **Decision Tree Classifier** | Performs strongly on Recall (83.69%) and AUC (0.8271) due to class weighting, but shows lower Precision (45.83%) and overall Accuracy (69.41%) compared to linear and ensemble models. | 
| **K-Nearest Neighbor Classifier** | Yields high overall Accuracy (76.72%) and decent Precision (56.65%), but struggles with Recall (52.41%) on high-dimensional feature spaces, missing roughly half of all actual churners. | 
| **Naive Bayes Classifier - Gaussian** | Achieves the highest overall Recall (86.90%) by flagging almost all potential churners. However, its strict independence assumption results in a high False Positive Rate, lowering its overall Accuracy to 65.37% and Precision to 42.54%. | 
| **Support Vector Machine** | Achieves the highest overall Accuracy (77.15%) while maintaining strong balanced metrics across AUC (0.8262), Precision (55.80%), Recall (66.84%), and F1 Score (0.6083), making it a competitive non-linear option. | 
| **Overall, Winner for the dataset** | **Logistic Regression** is the overall winner. It achieves the highest AUC Score (0.8486), highest F1 Score (0.6152), and highest MCC (0.4583) on unseen test data, providing the optimal business balance between capturing high-risk churners and avoiding unnecessary retention costs. |