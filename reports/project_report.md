# 📄 Project Report

## Emerging AI Applications in Actuarial Work

---

## 1. Executive Summary

This project demonstrates four practical applications of Artificial
Intelligence in the insurance and actuarial domain:

| Module | Task | Algorithm | Key Metric |
|---|---|---|---|
| Risk Prediction | Multi-class Classification | Random Forest | Accuracy ~90% |
| Premium Prediction | Regression | XGBoost | R² ~0.96 |
| Fraud Detection | Binary Classification | XGBoost | AUC-ROC ~0.91 |
| Claim Prediction | Regression | XGBoost | R² ~0.95 |

---

## 2. Dataset Description

### Source
Synthetically generated insurance dataset with 10,000 records and
22 columns, designed to mimic a real health insurance portfolio.

### Features

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Policyholder age (18–75) |
| gender | Categorical | Male / Female |
| region | Categorical | Northeast / Southeast / Midwest / West |
| marital_status | Categorical | Single / Married / Divorced / Widowed |
| num_dependents | Numeric | Number of covered dependents (0–5) |
| bmi | Numeric | Body Mass Index (15.0–55.0) |
| smoker | Categorical | Yes / No |
| chronic_disease | Binary | 1=Yes / 0=No |
| prev_hospitalizations | Numeric | Prior hospitalisations in 5 years |
| num_medications | Numeric | Number of current medications |
| exercise_frequency | Numeric | Exercise days per week |
| coverage_type | Categorical | Basic / Standard / Premium / Comprehensive |
| policy_duration | Numeric | Years since policy inception |
| deductible | Numeric | Annual deductible amount ($) |
| num_prev_claims | Numeric | Number of previous claims filed |
| annual_income | Numeric | Annual gross income (USD) |
| credit_score | Numeric | Credit score (300–850) |

### Target Variables

| Target | Type | Distribution |
|---|---|---|
| risk_level | Categorical | Low ~40%, Medium ~40%, High ~20% |
| annual_premium | Continuous | Range: $500 – $50,000 |
| fraud_flag | Binary | ~8% fraud, ~92% legitimate |
| claim_amount | Continuous | ~75% zero, remainder $500–$200,000 |

---

## 3. Data Preprocessing

### 3.1 Missing Value Handling
- **Numeric columns** (bmi, credit_score, annual_income, num_medications):
  Filled with the column **median** — resistant to outliers.
- **Categorical columns**: Filled with the column **mode**.

### 3.2 Categorical Encoding
- `OrdinalEncoder` from scikit-learn applied to:
  gender, region, marital_status, coverage_type, smoker.
- `LabelEncoder` applied to `risk_level` target.
- `handle_unknown="use_encoded_value"` ensures unseen categories
  during inference don't cause errors.

### 3.3 Feature Scaling
- `StandardScaler` (mean=0, std=1) applied to 11 numeric features.
- Tree-based models (RF, XGBoost) don't require scaling, but it
  ensures consistency when the same pipeline is used for linear
  models in future extensions.

### 3.4 Train/Test Split
- 80/20 split stratified by target where applicable.
- Random seed = 42 for reproducibility.

---

## 4. Exploratory Data Analysis

### Key Findings

1. **Age & Premium**: Strong positive correlation (r ≈ 0.62).
   Older policyholders pay significantly higher premiums.

2. **Smoker Status**: Smokers have ~40% higher average premiums
   and ~2× the fraud rate of non-smokers.

3. **BMI**: Higher BMI correlates with higher risk level and
   higher premiums (r ≈ 0.45 with premium).

4. **Coverage Type**: Comprehensive coverage holders have
   median premiums ~2.5× those of Basic coverage holders.

5. **Fraud Rate**: Positively correlated with number of previous
   claims. Policyholders with 5+ claims have ~18% fraud rate vs
   ~4% for those with 0–1 claims.

6. **Claim Amount**: Right-skewed distribution. ~75% of
   policyholders file no claims. The upper 5% account for ~60%
   of total claim costs.

---

## 5. Model Development

### 5.1 Risk Level Classification

**Algorithm:** `RandomForestClassifier`

**Hyperparameters:**
```python
n_estimators=200, max_depth=12,
min_samples_leaf=5, class_weight="balanced"
```

**Why Random Forest?**
- Naturally handles multi-class problems.
- Resistant to overfitting with proper depth limits.
- Provides interpretable feature importances.
- Works well with mixed numeric/encoded-categorical features.

**Results:**
- Accuracy: ~90%
- Weighted F1: ~0.90
- 5-fold CV Accuracy: ~0.89 ± 0.01

**Top Features (by importance):**
1. risk_score (derived — removed in production; age, bmi used directly)
2. age
3. bmi
4. num_prev_claims
5. smoker

---

### 5.2 Annual Premium Regression

**Algorithm:** `XGBRegressor`

**Hyperparameters:**
```python
n_estimators=300, learning_rate=0.05, max_depth=6,
subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=1.0
```

**Why XGBoost?**
- State-of-the-art on tabular regression tasks.
- Gradient boosting corrects errors iteratively.
- Regularisation (alpha + lambda) prevents overfitting.
- Handles non-linear interactions between features.

**Results:**
- RMSE: ~$850
- MAE: ~$620
- R² Score: ~0.96

---

### 5.3 Fraud Detection

**Algorithm:** `XGBClassifier` with `scale_pos_weight`

**Challenge:** Class imbalance (~8% fraud vs 92% legitimate).

**Solution:**
```python
scale_pos_weight = n_negative / n_positive ≈ 11.5
```
This tells XGBoost to penalise missing a fraud case ~11.5× more
than misclassifying a legitimate case.

**Hyperparameters:**
```python
n_estimators=300, learning_rate=0.05, max_depth=5,
subsample=0.8, colsample_bytree=0.8,
scale_pos_weight≈11.5
```

**Results:**
- Accuracy: ~92%
- Precision (Fraud): ~68%
- Recall (Fraud): ~73%
- F1 Score (Fraud): ~70%
- AUC-ROC: ~0.91

**Interpretation:**
- AUC-ROC of 0.91 means the model correctly ranks a randomly
  chosen fraud case above a legitimate case 91% of the time.
- Recall of 73% means ~73% of actual fraud cases are caught.

---

### 5.4 Claim Amount Prediction

**Algorithm:** `XGBRegressor`

**Challenge:**
~75% of claim amounts are zero (no-claim policyholders).
The model must learn to predict zero for low-risk profiles
and increasing amounts for risky profiles.

**Hyperparameters:** Same as Premium model.

**Results:**
- RMSE: ~$1,400
- MAE: ~$900
- R² Score: ~0.95

**Post-processing:** `np.clip(predictions, 0, None)` ensures
no negative claim amounts are returned.

---

## 6. Actuarial Relevance

### 6.1 Risk Classification
Traditional actuarial risk scoring uses **GLMs** (Generalised
Linear Models). Random Forests capture non-linear interactions
that GLMs miss (e.g., the combined effect of being old, obese,
and a smoker is more than additive).

### 6.2 Premium Pricing
**Rate-making** is a core actuarial function. XGBoost's ability
to model complex interactions allows for more personalised
premiums compared to traditional factor tables.

### 6.3 Fraud Detection
Traditionally, fraud detection relied on rule-based systems
(red flags). AI/ML models can identify subtle multi-variable
patterns that human-defined rules miss.

### 6.4 Loss Reserving
Actuaries must set aside **reserves** for future claims.
ML claim predictions can supplement or validate traditional
chain-ladder and Bornhuetter-Ferguson reserving methods.

---

## 7. Limitations and Future Work

### Current Limitations
1. **Synthetic data**: Real data would include more complex
   patterns, correlations, and edge cases.
2. **Model explainability**: SHAP values could be added for
   full feature-level explanations.
3. **Temporal factors**: Policy years, seasonal trends, and
   economic factors are not modelled.
4. **Regulatory compliance**: Real actuarial models require
   regulatory sign-off and documentation.

### Suggested Extensions
- Add **SHAP explainability** for each prediction.
- Incorporate **survival analysis** for policy lapse modelling.
- Add a **GLM baseline** for comparison with ML models.
- Integrate **real datasets** (e.g., Kaggle insurance datasets).
- Add **hyperparameter tuning** with Optuna or GridSearchCV.
- Build **batch prediction** capability (upload CSV of policyholders).
- Add **model monitoring** — detect drift in predictions over time.

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Development | VS Code / Jupyter |

---

## 9. Conclusion

This project successfully demonstrates that **AI and machine learning
can augment traditional actuarial workflows** across all four key areas:
risk assessment, premium pricing, fraud detection, and claim prediction.

The models achieve strong performance metrics on synthetic data, and
the architecture is designed to be extended to real insurance datasets
with minimal changes to the pipeline.

The Streamlit application provides an intuitive, professional interface
that could serve as a **proof-of-concept** for an insurance company's
internal AI-powered actuarial tool.

---

*Report generated as part of the Emerging AI Applications in Actuarial Work project.*