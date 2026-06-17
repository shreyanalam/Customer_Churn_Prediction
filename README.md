# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is
likely to cancel (churn) their subscription, based on their account,
billing, and service usage details. Built with Python, scikit-learn, and
an interactive Streamlit dashboard.



## 1. Project Overview

Customer churn (a customer leaving for a competitor) is one of the most
expensive problems for subscription-based businesses, so predicting it
early is a genuinely valuable, widely-used real-world application of
machine learning.

This project covers the full pipeline:

1. **Data Cleaning** -- handling missing values and irrelevant columns (`src/preprocess.py`)
2. **Visualization / EDA** -- understanding what drives churn through charts (`src/visualize.py`)
3. **Analysis** -- comparing churn rate across contract types, charges, tenure, etc.
4. **Prediction Models** -- training and comparing Logistic Regression and Random Forest classifiers (`src/train_model.py`)
5. **Dashboard Creation** -- a Streamlit app with a live prediction form and an analytics dashboard (`app.py`)

The dataset is a synthetically generated but realistically-structured
telecom customer dataset, created by `generate_dataset.py` so the whole
project runs end-to-end with zero external downloads.

---

## 2. Folder Structure
customer-churn-prediction/

├── data/

│   └── customer_churn_data.csv

├── models/

│   ├── churn_model.pkl

│   ├── scaler.pkl

│   ├── feature_columns.pkl

│   └── model_name.pkl

├── outputs/

│   ├── churn_distribution.png

│   ├── tenure_vs_churn.png

│   ├── contract_vs_churn.png

│   ├── monthly_charges_vs_churn.png

│   ├── correlation_heatmap.png

│   ├── confusion_matrix.png

│   ├── feature_importance.png

│   └── model_metrics.txt

├── src/

│   ├── __init__.py

│   ├── preprocess.py

│   ├── visualize.py

│   ├── train_model.py

│   └── predict_utils.py

├── app.py

├── generate_dataset.py

├── requirements.txt

├── .gitignore

└── README.md
---

## 3. Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

---

## 4. How to Run

```bash
python generate_dataset.py
python src/visualize.py
python src/train_model.py
streamlit run app.py
```

---

## 5. Expected Output

Training prints accuracy, precision, recall, F1, and ROC-AUC for both
models, and saves the best model to `models/`. The Streamlit app opens
two tabs: a live churn prediction form, and an analytics dashboard with
charts on churn rate by contract type, tenure, and monthly charges.

---

## 6. Project Explanation for Interview / Presentation

**What problem does it solve?** Predicting which customers are likely to
cancel their subscription so a business can proactively offer retention
incentives before losing the customer.

**Why two models?** Logistic Regression as an interpretable baseline,
Random Forest as the main model for handling non-linear relationships
and feature interactions.

**Why F1 score over accuracy?** Churn is a class-imbalanced problem
(fewer customers churn than stay), so accuracy alone can be misleading.

**What did the EDA reveal?** Month-to-month contracts, fiber-optic
internet without security/support add-ons, and electronic check payments
are associated with higher churn; longer tenure and contracts are
associated with lower churn.
