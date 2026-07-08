# Risk Alert Classifier — High-Risk Customer Prediction

A complete end-to-end machine learning project for a digital banking platform's early-warning system, built to identify high-risk customers (likely to default or commit fraud) from an imbalanced dataset of ~4,600 customers.

## Objective

Design, evaluate, and optimize a classification system that predicts high-risk customer behavior. The project implements multiple classification algorithms, evaluates them with advanced classification metrics, handles severe class imbalance, and improves performance using sampling techniques and hyperparameter tuning.

## Repository Structure

```
├── Risk_Alert_Classifier.ipynb   # Main Jupyter Notebook (fully executed, Parts A–H)
├── data/
│   └── risk_alert_dataset.csv    # Source dataset (4,600 customers)
├── THEORY_CONCEPTS.pdf           # Part A theory answers as a standalone document
├── README.md                     # This file
```

## Dataset

19 columns covering customer demographics, credit behavior, and transaction activity, with the target variable:

- `risk_status` → **0 = Low Risk**, **1 = High Risk** (highly imbalanced: ~88% / 12%)

Key features: `age`, `gender`, `region`, `employment_type`, `annual_income_inr`, `credit_score`, `credit_utilization_ratio`, `missed_payments_12m`, `avg_late_payment_days`, `monthly_transaction_count`, `monthly_spend_inr`, `cash_advance_count_6m`, `complaints_last_6m`, `failed_login_attempts_3m`, `account_tenure_months`, `last_transaction_date`, `debt_balance_inr`.

## Notebook Contents

| Part | Topic |
|------|-------|
| A | Conceptual Understanding (Logistic Regression, metrics, Type-I/II errors, AUC-ROC, imbalance theory) |
| B | Dataset understanding, stratified train-test split, KNN imputation |
| C | Baseline Logistic Regression, confusion matrix, Type-I/II error identification |
| D | Under-Sampling, Over-Sampling, SMOTE, ADASYN — before/after comparison |
| E | Decision Tree vs Random Forest, overfitting analysis |
| F | RandomizedSearchCV + GridSearchCV hyperparameter tuning |
| G | ROC curve analysis, AUC-ROC comparison, final model selection |
| H | Final report: best model, imbalance impact, metric comparison, business interpretation |

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
jupyter notebook Risk_Alert_Classifier.ipynb
```

Run all cells top to bottom — the notebook is self-contained and reads the dataset from `data/risk_alert_dataset.csv`.

##  Key Findings

- The dataset's class imbalance (~12% high-risk) causes the baseline Logistic Regression to have inflated accuracy but weak recall on the High-Risk class.
- SMOTE / ADASYN resampling substantially improves recall and AUC-ROC over the imbalanced baseline.
- Hyperparameter-tuned Random Forest reduces the overfitting seen in an untuned Decision Tree while maintaining strong recall.
- The final model is selected primarily on **recall** for the High-Risk class, since a missed high-risk customer (false negative) is the costlier business error for a bank's early-warning system.

## License / Use

Educational project for classification pipeline design and evaluation on imbalanced data.

## Author
Smit Patel
