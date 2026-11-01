# Telco Customer Churn Prediction & Model Evaluation Dashboard

## a. Problem Statement

Customer churn poses a significant challenge for telecommunication companies, directly impacting revenue and long-term business growth. The primary objective of this project is to develop, evaluate, and compare multiple Machine Learning classification models to accurately predict whether a customer will churn (leave the service) or remain retained based on their demographics, subscribed services, account details, and financial attributes. By establishing an automated batch prediction and evaluation dashboard, business teams can proactively identify high-risk churners and deploy targeted retention strategies.

## b. Dataset Description

* **Source & Provenance:** The dataset is sourced from the standard **IBM Sample Data Sets** (commonly hosted and benchmarked on **Kaggle** under *Telco Customer Churn*).
* **Original Dataset Size:** The full dataset contains **7,043 customer records** with **21 feature columns**.
* **Evaluation Test Subset (`test_data.csv`):** A held-out evaluation test set of **1,409 customer records** (80/20 split of the original dataset) was used for model validation.
  * **Target Variable:** `Churn` (`0` = Retained, `1` = Churned).
  * **Class Breakdown:** 1,035 Retained customers (73.46%) and 374 Churned customers (26.54%).

* **Feature Categories:**

  * **Demographics:** Gender, SeniorCitizen, Partner, Dependents.

  * **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.

  * **Account Info:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.

* **Preprocessing & Feature Alignment:** Categorical variables are numerical/one-hot encoded, and continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled using a standard scaler (`scaler.pkl`). Model inference uses a fixed probability threshold of **0.45**.

## c. Github Repository Link

https://github.com/2026aj05792/2026AJ05792_ML_Asgmt_2

## d. Models Used & Evaluation Comparison

### 1. Model Performance Comparison Table

The 5 classic classifiers were evaluated on the exact same test dataset (`1,409` records) with a fixed decision threshold of `0.45`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | 0.7715 | 0.8124 | 0.5598 | 0.6283 | 0.5921 | 0.4352 | 
| **Decision Tree Classifier** | 0.7480 | 0.6201 | 0.5556 | 0.2139 | 0.3089 | 0.2284 | 
| **K-Nearest Neighbor Classifier** | 0.7161 | 0.6908 | 0.4286 | 0.2487 | 0.3153 | 0.1784 | 
| **Naive Bayes Classifier - Gaussian** | 0.5947 | 0.7892 | 0.3888 | 0.8957 | 0.5421 | 0.3647 | 
| **Support Vector Machine** | 0.7679 | 0.8097 | 0.5564 | 0.6203 | 0.5866 | 0.4272 | 

### 2. Performance Observations per Model

| ML Model Name | Observation about model performance | 
| ----- | ----- | 
| **Logistic Regression** | Demonstrates the most balanced and consistent performance across all evaluated metrics. It achieved the highest overall Accuracy (77.15%), highest AUC (0.8124), highest F1 Score (0.5921), and highest MCC (0.4352), showing strong capability in separating churners from non-churners. | 
| **Decision Tree Classifier** | Shows weak generalization on unseen test data. While achieving decent accuracy (74.80%), its Recall (21.39%) and F1 Score (0.3089) are very low, indicating severe overfitting during training and an inability to detect high-risk churners. | 
| **K-Nearest Neighbor Classifier** | Performs poorly on this high-dimensional dataset (71.61% Accuracy, 0.1784 MCC). It heavily suffers from class imbalance, yielding a low Recall of 24.87% and missing nearly 75% of actual churners. | 
| **Naive Bayes Classifier - Gaussian** | Exhibits an extreme trade-off toward Recall (89.57%), correctly flagging 335 out of 374 actual churners. However, due to its strong independence assumption, it produces a high False Positive Rate, leading to the lowest overall Accuracy (59.47%) and lower Precision (38.88%). | 
| **Support Vector Machine** | Performs exceptionally well, closely trailing Logistic Regression as a top contender. It delivers solid stability across Accuracy (76.79%), AUC (0.8097), Precision (55.64%), and Recall (62.03%), making it a reliable linear classifier. | 
| **Overall, Winner for the dataset** | **Logistic Regression** is the overall winner. It yields the highest AUC (0.8124), Accuracy (77.15%), F1 Score (0.5921), and MCC (0.4352) at the 0.45 decision threshold, offering the ideal balance between catching churners and minimizing false alarms for production deployment. | 
