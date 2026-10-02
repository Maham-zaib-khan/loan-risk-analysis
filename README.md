# loan-risk-analysis
This project demonstrates a complete loan data preprocessing workflow including data cleaning, missing value handling, duplicate removal, outlier detection, feature engineering, categorical encoding, feature scaling and build a ready to use SVM classification optimized model
# 🏦 Retail Loan Risk Assessment & Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-orange)
![Framework](https://img.shields.io/badge/Deployed%20with-Streamlit-FF4B4B)

## 📌 Project Overview
This project is an end-to-end machine learning pipeline and interactive web application designed for retail banks. It analyzes borrower financial metrics to predict the likelihood of loan default. By shifting from standard underwriting rules to a data-driven Support Vector Machine (SVM) model, this tool helps financial institutions minimize charge-offs while maintaining healthy origination volume.

## 🚀 Features
* **Automated Data Cleaning & Imputation:** Intelligently handles missing values (e.g., median imputation for missing incomes) and removes extreme outliers.
* **Custom Feature Engineering:** Calculates the **Loan-to-Income (LTI) ratio** dynamically, which Exploratory Data Analysis (EDA) proved to be a critical indicator of default risk.
* **Optimized Machine Learning Model:** Utilizes a Support Vector Machine (SVM) optimized via `GridSearchCV` to balance precision and recall, utilizing class weighting to handle imbalanced default data.
* **Interactive Web UI:** Deployed as a lightweight, user-friendly web app using Streamlit, allowing loan officers to input applicant data and receive instant risk classifications.

## 🛠️ Tech Stack
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (SVC, StandardScaler, Pipeline, GridSearchCV)
* **Deployment:** `streamlit`, `joblib`

## 📁 Repository Structure
```text
├── data/
│   └── loan_data.csv          # Original dataset (Not uploaded due to privacy)
├── notebooks/
│   └── eda_and_modeling.ipynb # Jupyter notebook containing full EDA and model training
├── models/
│   └── svm_loan_model.pkl     # Exported SVM Pipeline ready for deployment
├── app.py                     # Streamlit web application script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
