# Hybrid Ensemble Model Development Summary

## Overview

Successfully created a hybrid ensemble model that combines traditional Home Credit features with behavioral UCI Credit Card features for enhanced loan default prediction.

## Models Combined

1. **model_hybrid.pkl** - Traditional Home Credit model (487 features)
2. **first_lgbm_model.pkl** - Behavioral UCI Credit Card model (31 features)
3. **model_ensemble_wrapper.pkl** - NEW Ensemble meta-learner (combines both)

## Feature Engineering

### Created Files

1. **src/create_hybrid_features.py** - Feature simulation script
2. **src/train_ensemble_hybrid.py** - Ensemble training script

### Generated Datasets

1. **data/smoke_hybrid_features.csv**

   - 20,000 rows x 527 features
   - Original Home Credit features + Simulated behavioral features
   - Includes TARGET column (17,339 non-null values)

2. **data/uci_hybrid_features.csv**
   - 1,425 rows x 57 features
   - Original UCI features + Simulated traditional features

### Simulated Features

#### Behavioral Features for Home Credit Users (39 features)

- **Original UCI Features**: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
- **Payment Status**: PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
- **Bill Amounts**: BILL_AMT1-6 (simulated from AMT_CREDIT and AMT_INCOME_TOTAL)
- **Payment Amounts**: PAY_AMT1-6 (simulated from AMT_ANNUITY)
- **Engineered Features**:
  - total_billed_amount, total_payment_amount
  - avg_transaction_amount, max_billed_amount, max_payment_amount
  - spending_volatility, income_consistency
  - rolling_balance_volatility, net_flow_balance
  - debt_stress_index, repayment_ratio
  - payment_consistency_ratio, spend_to_income_volatility_ratio
  - max_to_mean_bill_ratio, missed_payment_count
  - credit_utilization_trend

#### Traditional Features for UCI Users (24 features)

- DAYS_BIRTH, DAYS_EMPLOYED, AMT_INCOME_TOTAL
- AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE
- EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
- CNT_FAM_MEMBERS, OWN_CAR_AGE
- Engineered ratio features (APPS\_\*)

## Ensemble Model Architecture

### Stacking Approach

```
Input Data
    ↓
    ├─→ Traditional Model (model_hybrid.pkl) → pred_traditional
    └─→ Behavioral Model (first_lgbm_model.pkl) → pred_behavioral
                ↓
         Meta-Features (7 features):
         - pred_traditional
         - pred_behavioral
         - pred_avg (average)
         - pred_max (maximum)
         - pred_min (minimum)
         - pred_diff (absolute difference)
         - pred_ratio (traditional/behavioral)
                ↓
         LightGBM Meta-Learner
                ↓
         Final Prediction
```

### Model Performance

- **AUC-ROC**: 0.8591
- **Accuracy**: 93%
- **Precision**: 0.66 (class 1)
- **Recall**: 0.09 (class 1)
- **F1-Score**: 0.16 (class 1)

**Note**: Model is conservative with high precision but low recall for defaults (class 1). This means it's accurate when it predicts default but misses many actual defaults.

### Confusion Matrix

```
Predicted:   No Default  |  Default
Actual:
No Default      3185      |    13
Default          245      |    25
```

## Saved Models

### 1. model_ensemble_hybrid.pkl

- Raw LightGBM meta-learner
- Requires manual feature preparation
- Use for custom pipelines

### 2. model_ensemble_wrapper.pkl (RECOMMENDED)

- Complete wrapper class with preprocessing
- Handles missing values and categorical encoding
- Ready-to-use with single function call
- Usage:

```python
import joblib
wrapper = joblib.load('models/model_ensemble_wrapper.pkl')
predictions = wrapper.predict(X)
probabilities = wrapper.predict_proba(X)
```

### 3. ensemble_metadata.pkl

- Feature lists for both models
- Model paths
- Ensemble configuration

## Next Steps

### 1. Update Streamlit App

- Add "Ensemble Hybrid Model" option to model selection
- Update prediction page to handle hybrid features
- Add explanation of ensemble predictions

### 2. Potential Improvements

- **Address Class Imbalance**:
  - Use SMOTE or other oversampling techniques
  - Adjust class weights in meta-learner
  - Try different thresholds for classification
- **Feature Selection**:
  - Remove redundant features
  - Add more key features to meta-learner
- **Model Tuning**:
  - Hyperparameter optimization for meta-learner
  - Try different ensemble strategies (voting, weighted average)
- **Cross-Validation**:
  - Implement stratified k-fold CV
  - Validate on completely held-out data

### 3. Production Deployment

- Create API endpoint for ensemble predictions
- Add monitoring for model drift
- Implement A/B testing framework
- Document feature requirements

## File Structure

```
Loan Default Hybrid System/
├── src/
│   ├── create_hybrid_features.py    ← Feature simulation
│   ├── train_ensemble_hybrid.py     ← Ensemble training
│   └── feature_engineering.py       ← Updated with behavioral features
├── models/
│   ├── model_hybrid.pkl              ← Traditional model
│   ├── first_lgbm_model.pkl          ← Behavioral model
│   ├── model_ensemble_hybrid.pkl     ← NEW: Meta-learner
│   ├── model_ensemble_wrapper.pkl    ← NEW: Ready-to-use wrapper
│   └── ensemble_metadata.pkl         ← NEW: Metadata
├── data/
│   ├── smoke_engineered.csv          ← Original data
│   ├── smoke_hybrid_features.csv     ← NEW: With behavioral features
│   ├── uci_interface_test.csv        ← Original UCI data
│   └── uci_hybrid_features.csv       ← NEW: With traditional features
└── HYBRID_MODEL_SUMMARY.md          ← This file
```

## Author

Daniel Ajayi

## Date

November 11, 2025
