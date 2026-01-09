# Loan Default Risk Prediction

This project builds and evaluates machine learning models to predict the probability of loan default using the Give Me Some Credit dataset (Kaggle).

The goal is to simulate an industry-style credit risk modeling workflow, including:
- baseline logistic regression
- gradient boosting (XGBoost)
- policy-based evaluation using top-k risk thresholds
- model governance artifacts (ROC, PR, confusion matrices, metrics tables)

---

## Dataset
Source: https://www.kaggle.com/competitions/GiveMeSomeCredit

Target:
- `SeriousDlqin2yrs` (1 = default, 0 = no default)

---

## Models

### Baseline: Logistic Regression
- Median imputation
- Log transform on skewed features
- Standard scaling
- Class-weighted loss

### Final Model: XGBoost
- Tree-based gradient boosting
- Tuned via randomized search
- Evaluated with AUC, PR, and policy-based metrics

---

## Results

| Model | Validation AUC |
|------|----------------|
| Logistic Regression | ~0.86 |
| XGBoost | ~0.87 |

### Policy Evaluation (Top 5% Review Capacity)

| Model | Precision | Recall |
|------|------------|--------|
| Logistic Regression | 0.435 | 0.325 |
| XGBoost | 0.496 | 0.371 |

XGBoost captures more defaulters while wasting fewer reviews at the same operational capacity.

---

## Artifacts

Saved to `artifacts/`:
- trained model
- ROC and PR curves
- confusion matrices (2% and 5% policy)
- top-k threshold metrics

---

## How to Run

```bash
pip install -r requirements.txt
python src/train_xgb.py
