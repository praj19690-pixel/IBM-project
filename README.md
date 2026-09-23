# AI-Powered Customer Churn Prediction & Retention Analytics

**Student:** Priyanshu Raj  
**Internship:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026  
**Dataset:** Telco Customer Churn (Kaggle)  
**Kaggle URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Project Overview

This project combines **Data Analytics**, **Exploratory Data Analysis (EDA)**, **Machine Learning**, **Model Interpretability**, **AI-Based Insights**, and an interactive **Streamlit Dashboard** to analyse customer churn patterns in a telecommunications company and predict which customers are at risk of churning.

---

## Problem Statement

> "Can customer account, service, billing, and usage characteristics be analysed to understand churn patterns and build a machine-learning model that predicts whether a customer is likely to churn?"

---

## Objectives

1. Inspect and clean the Telco Customer Churn dataset.
2. Perform comprehensive EDA to identify patterns associated with churn.
3. Engineer meaningful features from the available data.
4. Calculate dynamic KPIs (churn rate, average tenure, charges, etc.).
5. Train and evaluate multiple classification models (Logistic Regression, Random Forest, Gradient Boosting).
6. Interpret model behaviour via feature importance.
7. Generate automated AI-based analytical insights from actual computed results.
8. Deliver an interactive Streamlit dashboard with sidebar filters, prediction form, and 8 pages.

---

## Dataset

| Property | Value |
|---|---|
| Name | Telco Customer Churn |
| Source | IBM Sample Data / Kaggle |
| URL | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Filename | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes / No) |

---

## Dataset Setup

> The raw CSV is **not included** in this repository. Download it manually:

1. Visit https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click **Download** (requires free Kaggle account)
3. Extract and place the file at:

```
PriyanshuRaj_CustomerChurnAnalytics/
└── data/
    └── raw/
        └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The processed file (`data/processed/telco_churn_cleaned.csv`) is generated automatically on first run.

---

## Project Structure

```
PriyanshuRaj_CustomerChurnAnalytics/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← place here (not in git)
│   └── processed/
│       └── telco_churn_cleaned.csv                 ← auto-generated
│
├── screenshots/                                     ← capture after running
│
├── PriyanshuRaj_CustomerChurnAnalytics.py           ← main Streamlit app
├── PriyanshuRaj_CustomerChurnAnalytics_ProjectReport.docx
├── requirements.txt
└── README.md
```

---

## Data Dictionary

| Column | Type | Description |
|---|---|---|
| customerID | str | Unique customer ID (dropped before ML) |
| gender | str | Customer gender: Male / Female |
| SeniorCitizen | int→str | Senior citizen: 1=Yes, 0=No |
| Partner | str | Has a partner: Yes / No |
| Dependents | str | Has dependents: Yes / No |
| tenure | int | Months with company (0–72) |
| PhoneService | str | Phone service: Yes / No |
| MultipleLines | str | Multiple phone lines |
| InternetService | str | Internet type: DSL / Fiber optic / No |
| OnlineSecurity | str | Online security add-on |
| OnlineBackup | str | Online backup add-on |
| DeviceProtection | str | Device protection add-on |
| TechSupport | str | Tech support add-on |
| StreamingTV | str | Streaming TV add-on |
| StreamingMovies | str | Streaming movies add-on |
| Contract | str | Contract: Month-to-month / One year / Two year |
| PaperlessBilling | str | Uses paperless billing: Yes / No |
| PaymentMethod | str | Payment method (4 types) |
| MonthlyCharges | float | Monthly bill in USD |
| TotalCharges | float | Total billed to date in USD |
| **Churn** | str | **Target: Yes / No** |

---

## Data Cleaning Summary

| Issue | Fix |
|---|---|
| `TotalCharges` stored as string | Cast to float64 |
| 11 blank `TotalCharges` (tenure=0) | Set to 0.0 |
| `SeniorCitizen` as int 0/1 | Mapped to No/Yes |
| `customerID` — identifier only | Dropped before ML |

---

## Feature Engineering

| Feature | Logic | Purpose |
|---|---|---|
| `TenureGroup` | Bins: 0-12, 13-24, 25-48, 49-72 months | EDA grouping |
| `ChargeBand` | Bins: Low/Medium/High/Very High | EDA grouping |
| `ServiceCount` | Count of 6 internet add-on subscriptions | ML feature |
| `HasPhone` | 1 if PhoneService=Yes | ML feature |
| `HasInternet` | 1 if InternetService != No | ML feature |
| `ChargePerTenure` | TotalCharges / tenure (or MonthlyCharges if tenure=0) | ML feature |

---

## Churn Analysis (actual computed values)

| Metric | Value |
|---|---|
| Total Customers | 7,043 |
| Churned | 1,869 |
| Retained | 5,174 |
| Churn Rate | **26.54%** |
| Avg Tenure (churned) | **17.98 months** |
| Avg Tenure (retained) | **37.57 months** |
| Avg Monthly Charges (churned) | **$74.44** |
| Avg Monthly Charges (retained) | **$61.27** |

---

## Machine Learning

### Models Trained
- Logistic Regression
- Random Forest (200 estimators)
- Gradient Boosting (200 estimators)

### Pipeline
- 80/20 train/test split, stratified on target, `random_state=42`
- `ColumnTransformer`: StandardScaler (numeric) + OneHotEncoder (categorical)
- Sklearn `Pipeline` used for both training and prediction

### Model Results (actual computed, test set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80.62% | 60.61% | **84.18%** |
| Random Forest | 78.64% | 54.74% | 82.22% |
| Gradient Boosting | 79.35% | 56.24% | 83.85% |

**Best model by ROC-AUC:** Logistic Regression (84.18%)

---

## Dashboard Pages

| Page | Description |
|---|---|
| Executive Overview | KPI cards, churn donut, contract/tenure charts |
| Churn Analysis | Churn by service, payment, demographics |
| Customer & Service Analysis | Tenure, charges, service count charts |
| Machine Learning | Model comparison, confusion matrices, ROC curves |
| Churn Prediction | Interactive input form with probability gauge |
| Feature Importance | Top 15 features per model |
| AI Insights | Automated insights from actual results |
| Data Quality | Cleaning report, target distribution |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run PriyanshuRaj_CustomerChurnAnalytics.py
```

> Make sure the raw CSV is placed at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` first.

---

## Technologies Used

| Library | Purpose |
|---|---|
| pandas | Data loading, cleaning, analysis |
| numpy | Numeric operations |
| scikit-learn | ML models, preprocessing, evaluation |
| plotly | Interactive visualisations |
| streamlit | Dashboard framework |
| python-docx | Project report generation |

---

## Key Findings

1. **Month-to-month contracts** show the highest observed churn rate.
2. **Shorter tenure** (0–12 months) is associated with higher observed churn.
3. **Fiber optic internet** customers show a higher observed churn rate than DSL customers.
4. **Electronic check** payment is associated with a higher observed churn rate vs. automatic methods.
5. **Churned customers** pay ~$13/month more on average and have ~20 fewer months of tenure.
6. **Logistic Regression** achieved the best ROC-AUC (84.18%) on the held-out test set.

---

## Limitations

- Dataset is a single snapshot in time; no temporal sequence is modelled.
- Observed associations do not imply causation.
- Model performance is bounded by the features available in the dataset.
- Predictions are probabilistic and should not be treated as certain outcomes.

---

## Future Improvements

- Incorporate SHAP for per-prediction explainability.
- Experiment with XGBoost / LightGBM.
- Add time-series churn modelling if longitudinal data becomes available.
- Deploy as a cloud-hosted Streamlit app.

---

## Ethical Considerations

- No personally identifiable information is used in model training.
- Model predictions are presented as probabilities, not certainties.
- Causal language is avoided throughout.
- No external paid APIs or data sources are used.

---

## Dataset Attribution

IBM Telco Customer Churn dataset, sourced from Kaggle:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Author

**Priyanshu Raj**  
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
