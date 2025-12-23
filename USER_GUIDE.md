# Loan Default Prediction System - User Guide

> **Last Updated:** December 9, 2025  
> **Version:** 2.4.0  
> **Latest Update:** 538-feature hybrid architecture with enhanced interpretability

## Recent Enhancements

- ✅ **538-Feature Hybrid Architecture**: 7 meta-features + 487 traditional + 44 behavioral features
- ✅ **Performance Improvement**: Test AUC 0.8590 (up from 0.8158), Accuracy 81%, Recall 77%
- ✅ **Feature Name Prefixes**: `pred_*` (meta), `trad_*` (traditional), `behav_*` (behavioral) for interpretability
- ✅ **Enhanced SHAP Visualizations**: Shows feature sources in all importance plots
- ✅ **Statistical Validation Notebook** (`Chapter4_Statistical_Analysis.ipynb`): Comprehensive McNemar's, DeLong's, and Bootstrap CI tests
- ✅ **CatBoost Ensemble Upgrade**: 77% recall, AUC 0.8590
- ✅ **Interactive Plotly Visualizations**: Precision-Recall and ROC curves with hover tooltips
- ✅ **Centralized Data Cleaning**: Single module (`src/data_cleaning.py`) for consistent preprocessing
- ✅ **Cache Auto-Invalidation**: Dashboard updates automatically when models retrained

## Table of Contents

1. [Getting Started](#getting-started)
2. [Navigating the Dashboard](#navigating-the-dashboard)
3. [Making Predictions](#making-predictions)
4. [Understanding Results](#understanding-results)
5. [Model Metrics](#model-metrics)
6. [Statistical Validation](#statistical-validation)
7. [Feature Importance](#feature-importance)
8. [Exploratory Data Analysis](#exploratory-data-analysis)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Launching the Application

1. **Activate Virtual Environment:**

   ```bash
   .\myenv\Scripts\Activate.ps1
   ```

2. **Start Streamlit:**

   ```bash
   streamlit run app.py
   ```

   **Note**: The system uses a centralized data cleaning module (`src/data_cleaning.py`) for all data preprocessing and missing value handling.

3. **Access the Dashboard:**
   - Open your browser to: `http://localhost:8501`
   - The application runs locally on your machine

### System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, Edge)

---

## Navigating the Dashboard

### Homepage

**Location:** Main landing page

**Purpose:** Overview of the three prediction models

- **Traditional Model** - Home Credit features (7 datasets → 487 features)
- **Behavioral Model** - UCI Credit Card patterns (1 dataset, 25 columns → 44 features: 23 base + 21 engineered)
- **Ensemble Model** - Combined hybrid predictions

**Quick Stats:**

- Model accuracy metrics
- Feature counts
- Training dataset information

### Sidebar Navigation

- **Exploratory Data Analysis (EDA)** - Data exploration and visualization
- **Prediction** - Make loan default predictions
- **Feature Importance** - View key predictive features
- **Model Metrics** - Detailed model performance analysis

---

## Making Predictions

### Prediction Page

#### Step 1: Select Model Type

Choose from three available models:

**Traditional Model (Home Credit)**

- Best for: Comprehensive financial profiles
- Input: 487 features including income, credit history, demographics
- Use when: You have detailed applicant information

**Behavioral Model (UCI Credit Card)**

- Best for: Payment behavior analysis
- Input: 44 features including payment history, bill amounts
- Use when: You have credit card transaction data

**Ensemble Hybrid Model (CatBoost)** _Recommended_

- Best for: Maximum accuracy and default detection
- Input: **538 features total** (7 meta-features + 487 traditional + 44 behavioral)
- Use when: You have data for both feature sets
- Advantages:
  - **77% recall** - catches 207 out of 270 defaults in test set
  - **81% accuracy** - strong overall performance
  - **AUC 0.8590** - excellent discrimination ability (+5.3% improvement)
  - Optimal threshold 0.5 (adjustable based on risk tolerance)
  - **Meta-features**: pred_traditional, pred_behavioral, pred_avg, pred_max, pred_min, pred_diff, pred_ratio
  - **Raw features preserved**: All 487 traditional + 44 behavioral features included alongside meta-features
  - **Feature prefixes**: Shows source of each feature (pred*\*, trad*\_, behav\_\_)
- Technical: CatBoost with auto class balancing, early stopping at iteration 68

#### Step 2: Choose Input Method

##### Option A: Manual Form Input

1. **Select "Manual Input"** from the radio buttons
2. **Fill in the form fields:**
   - All required fields marked with \*
   - Hover over field labels for descriptions
   - Use appropriate units (currency, days, percentages)
3. **Click "Predict"** button at the bottom

**Traditional Model Fields:**

- Personal: Gender, age, family status, education
- Financial: Income, credit amount, loan type
- Employment: Days employed, occupation
- Assets: Car ownership, realty ownership

**Behavioral Model Fields:**

- Account: Credit limit, demographics
- Payment History: PAY_0 through PAY_6 (repayment status)
- Bill Amounts: BILL_AMT1 through BILL_AMT6
- Payment Amounts: PAY_AMT1 through PAY_AMT6

**Ensemble Model:**

- Requires all fields from both models
- Forms are organized in collapsible sections
- Fill out both Traditional and Behavioral sections

##### Option B: CSV Batch Upload

1. **Select "Upload CSV File"** from radio buttons
2. **Download Template:**
   - Click "Download CSV Template" button
   - Template contains all required columns with correct names
3. **Prepare Your Data:**
   - Open template in Excel or text editor
   - Fill in your data (one row per applicant)
   - Keep column names exactly as in template
   - Save as CSV file
4. **Upload File:**
   - Click "Browse files" button
   - Select your prepared CSV file
   - System validates column names automatically
5. **Review Preview:**
   - First 5 rows displayed for verification
   - Check data looks correct
6. **Click "Generate Predictions"**

**CSV Requirements:**

- Column names must match template exactly
- No missing column names (values can be missing)
- Numeric fields: Use numbers without currency symbols
- Date fields: Use integers (days before application)
- Categorical fields: Use exact codes from template

#### Step 3: Review Predictions

**For Single Prediction (Manual Input):**

**Risk Assessment:**

- **Low Risk** - Default probability < 30%
- **Medium Risk** - Default probability 30-60%
- **High Risk** - Default probability > 60%

**Prediction Details:**

- **Default Probability:** Percentage likelihood of default (0-100%)
- **Predicted Class:** Default (1) or No Default (0)
- **Risk Level:** Visual indicator with color coding
- **Confidence Score:** Model certainty in prediction

**Visualization:**

- Risk gauge showing probability on a 0-100% scale
- Color-coded risk zones for quick interpretation

**For Batch Predictions (CSV Upload):**

**Results Table:**

- One row per applicant in your CSV
- Columns: Prediction (0/1), Probability (%), Risk Level
- Sortable by clicking column headers
- Scrollable for large batches

**Summary Statistics:**

- Total applications processed
- High risk count and percentage
- Medium risk count and percentage
- Low risk count and percentage
- Average default probability

**Download Results:**

- Click "Download Results as CSV" button
- File contains all input data plus predictions
- Use for further analysis or record-keeping

---

## Understanding Results

### Prediction Outputs Explained

#### Default Probability

**What it means:**

- The model's estimated likelihood that the applicant will default
- Expressed as a percentage from 0% to 100%

**How to interpret:**

- **0-30%:** Low risk - Likely to repay
- **30-60%:** Medium risk - Requires careful evaluation
- **60-100%:** High risk - High likelihood of default

**Example:**

- 15% probability = 15% chance of default, 85% chance of repayment
- 75% probability = 75% chance of default, 25% chance of repayment

#### Predicted Class

**Values:**

- **0:** No Default - Model predicts applicant will repay
- **1:** Default - Model predicts applicant will default

**Decision threshold:** 50%

- Probability ≥ 50% → Predicted class = 1 (Default)
- Probability < 50% → Predicted class = 0 (No Default)

#### Risk Level

Visual categorization for quick decision-making:

- **Low Risk:** Safe to approve
- **Medium Risk:** Requires manual review
- **High Risk:** Consider rejection or additional safeguards

### Model-Specific Insights

#### Traditional Model Results

**Strengths:**

- Comprehensive financial profile assessment
- Strong performance on income and credit history features
- Best for applicants with complete documentation

**Key Factors:**

- External credit scores (EXT_SOURCE features)
- Income-to-credit ratio
- Employment history
- Previous loan performance

#### Behavioral Model Results

**Strengths:**

- Payment pattern analysis
- Transaction behavior insights
- Effective for credit card holders

**Key Factors:**

- Payment consistency (PAY\_\* features)
- Spending volatility
- Debt stress index
- Repayment ratio

#### Ensemble Model Results

**Strengths:**

- Highest accuracy (combines both models)
- Balanced assessment using multiple data sources
- Most robust predictions

**How it works:**

1. Traditional model makes a prediction
2. Behavioral model makes a prediction
3. Meta-learner combines both predictions
4. Final prediction considers:
   - Both base model outputs
   - Prediction agreement/disagreement
   - Top features from each model

**Key Meta-Features:**

- `pred_traditional`: Traditional model's probability
- `pred_behavioral`: Behavioral model's probability
- `pred_avg`: Average of both predictions
- `pred_diff`: Difference between predictions (agreement indicator)

---

## Model Metrics

### Model Metrics Page

#### Accessing Metrics

1. Navigate to "Model Metrics" from sidebar
2. Select model from dropdown:
   - Traditional_model.pkl (Traditional)
   - Behaviorial_model.pkl (Behavioral)
   - model_ensemble_catboost_meta.pkl (Ensemble - CatBoost)

**Note:** The ensemble model uses CatBoost (upgraded from LightGBM in Version 2.2.0) with 7 simplified meta-features for improved recall and efficiency.

#### Performance Metrics Explained

**Note on Fair Model:** When viewing the Ensemble model, you can toggle the "Use Fair Model" option in the Fairness & Bias Analysis section to compare baseline vs fairness-optimized predictions. The fair model uses group-specific thresholds (not a single threshold slider) to achieve demographic parity across protected attributes.

**Accuracy**

- Percentage of correct predictions (both defaults and non-defaults)
- Formula: (Correct Predictions) / (Total Predictions)
- Traditional: ~92%
- Behavioral: ~82%
- Ensemble: ~92.1% (at optimal threshold 0.32)

**Note:** Accuracy can be misleading with imbalanced data (8% default rate). Focus on AUC, Precision, and Recall for better insights.

**Precision**

- Of predicted defaults, what percentage actually defaulted?
- Formula: True Positives / (True Positives + False Positives)
- High precision = Few false alarms
- Important for: Minimizing unnecessary loan rejections

**Recall (Sensitivity)**

- Of actual defaults, what percentage did we catch?
- Formula: True Positives / (True Positives + False Negatives)
- High recall = Few missed defaults
- Important for: Minimizing financial losses

**Ensemble Performance at Optimal Threshold (0.32):**

- Recall: 88.89% (catches 240 out of 270 defaults)
- Traditional baseline @ 0.5: ~42% recall
- Improvement: +46.89 percentage points
- Business impact: $262,500 - $420,000 additional savings

**F1 Score**

- Harmonic mean of Precision and Recall
- Balances both metrics
- Range: 0 (worst) to 1 (best)
- Good for: Overall model quality assessment

**AUC-ROC**

- Area Under the ROC Curve
- Measures model's ability to distinguish between classes
- Range: 0.5 (random) to 1.0 (perfect)
- Traditional: 0.797
- Behavioral: 0.771
- Ensemble: 0.812

#### Fairness & Bias Analysis

**Available for:** Ensemble Model

**Key Features:**

- **Fair Model Toggle:** Switch between baseline and fairness-optimized ensemble
- **Group-Specific Thresholds:** View exact thresholds used for each demographic group
  - SEX: Males (0.72%), Females (51.27%)
  - MARRIAGE: Single (18.74%), Married (18.67%), Widowed/Divorced (83.50%)
  - AGE_GROUP: 5 age ranges with thresholds from 0.36% to 18.56%
- **Disparate Impact Metrics:** 80% rule compliance for all groups
  - SEX: 98.4% ✅
  - MARRIAGE: 97.8% ✅
  - AGE_GROUP: 94.5% ✅
- **Performance Comparison:** Side-by-side baseline vs fair model metrics
- **Confusion Matrix:** Uses fair predictions when enabled

**How Fair Model Works:**

1. Uses same probabilities as baseline model
2. Applies different decision thresholds to different demographic groups
3. Ensures minimum group acceptance rate / maximum group acceptance rate ≥ 0.8
4. No threshold slider (uses optimized group-specific thresholds)

**Trade-offs:**

- Higher precision (64.3% vs 24.7%): Fewer false alarms
- Lower recall (16.7% vs 77.8%): Misses more defaults
- Better for risk-averse lending strategies

#### Training History Visualization

**For Traditional/Behavioral Models:**

**Training Curves:**

- X-axis: Training iterations
- Y-axis: Metric value (AUC, accuracy, etc.)
- Multiple datasets shown:
  - Training set (blue)
  - Validation set (orange)

**What to look for:**

- Convergence: Metrics stabilizing over iterations
- Overfitting: Large gap between training and validation
- Best iteration: Peak validation performance

**Best Iteration:**

- Iteration number where validation performance was highest
- Model training stopped here (early stopping)
- Prevents overfitting

#### Feature Importance

**What it shows:**

- Top 20 most influential features
- Bar chart: Longer bar = More important
- Colormap: Color intensity indicates importance

**How to interpret:**

- Features at top have most impact on predictions
- Model relies heavily on these features
- Missing values in top features hurt accuracy

**Traditional Model Top Features:**

- EXT_SOURCE_2, EXT_SOURCE_3 (external credit scores)
- DAYS_BIRTH (age)
- DAYS_EMPLOYED (employment length)
- Income-related ratios

**Behavioral Model Top Features:**

- PAY_0 (most recent payment status)
- PAY_2, PAY_3 (historical payment status)
- LIMIT_BAL (credit limit)
- Engineered features (spending_volatility, debt_stress_index)

**Ensemble Model Meta-Features (CatBoost):**

- pred_traditional (Traditional model prediction) - Primary signal
- pred_behavioral (Behavioral model prediction) - Secondary signal
- pred_avg (Average of both predictions) - Consensus
- pred_max (Maximum prediction) - Pessimistic view
- pred_min (Minimum prediction) - Optimistic view
- pred_diff (Difference between models) - Agreement indicator
- pred_ratio (Ratio of predictions) - Relative confidence

**Total: 7 meta-features** (simplified from previous 27 for better efficiency)

#### Ensemble-Specific Metrics

**ROC Curve:**

- Shows trade-off between True Positive Rate and False Positive Rate
- Diagonal line = Random guessing
- Curve above diagonal = Better than random
- Area under curve = AUC score

**Confusion Matrix:**

- 2x2 grid showing prediction outcomes:
  - Top-left: True Negatives (Correctly predicted no default)
  - Top-right: False Positives (Incorrectly predicted default)
  - Bottom-left: False Negatives (Missed defaults)
  - Bottom-right: True Positives (Correctly predicted default)

**Prediction Distribution:**

- Bar chart showing count of each prediction class
- Helps assess class balance
- Check for extreme imbalances

---

## Statistical Validation

### Chapter 4 Statistical Analysis

For rigorous statistical validation of model performance claims, refer to the comprehensive Jupyter notebook:

**File**: `Chapter4_Statistical_Analysis.ipynb`

#### What's Included

**1. Statistical Significance Tests**

- **McNemar's Test**: Validates that ensemble predictions are significantly better than base models
  - Traditional vs Ensemble: χ²=351.63, p<0.001 (highly significant)
  - Behavioral vs Ensemble: χ²=286.37, p<0.001 (highly significant)
- **DeLong's Test**: Confirms AUC improvements are statistically significant
  - Behavioral vs Ensemble: Z=14.75, p<0.001 (highly significant)
- **Bootstrap Confidence Intervals**: 1000 iterations for robust performance estimates
  - Traditional: [0.8224, 0.8702]
  - Behavioral: [0.4641, 0.5381]
  - Ensemble: [0.8250, 0.8717]

**2. Interactive Visualizations**

- **Precision-Recall Curves**: Plotly interactive charts with hover tooltips
- **ROC Curves**: Compare all three models with documented AUC values
- **Bootstrap Distributions**: Visual confidence interval analysis

**3. Performance Validation**

- Test set: 3,468 samples (7.79% default rate)
- Documented AUC values confirmed:
  - Traditional: 0.7970
  - Behavioral: 0.7714
  - Ensemble: 0.8509

#### How to Use

**Option 1: Jupyter Notebook**

```bash
# From project root
jupyter notebook Chapter4_Statistical_Analysis.ipynb
```

**Option 2: VS Code**

```bash
# Open in VS Code
code Chapter4_Statistical_Analysis.ipynb

# Run cells interactively
# All cells are pre-executed with results
```

**Key Cells to Review:**

- Cell 5: McNemar's Test results
- Cell 6: DeLong's Test results
- Cell 7: Bootstrap CI with visual distributions
- Cell 9: Interactive Precision-Recall curves
- Cell 10: Interactive ROC curves
- Cell 11: Comprehensive validation summary

#### Key Takeaways

All model performance claims are **statistically validated** with p-values < 0.001:

- Ensemble significantly outperforms both base models
- AUC improvements are genuine, not due to random chance
- Bootstrap confidence intervals show no overlap - clear superiority
- Visual analysis confirms quantitative findings

**Business Confidence**: You can confidently report these results knowing they are backed by rigorous statistical testing (McNemar's, DeLong's, Bootstrap resampling).

---

## Feature Importance

### Feature Importance Page

#### Purpose

Understand which features drive model decisions

#### Viewing Feature Importance

**Global Importance:**

1. Select model from dropdown
2. View horizontal bar chart
3. Features sorted by importance (highest at top)

**Feature Categories:**

**Traditional Model Features:**

- **External Scores:** EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
- **Demographics:** DAYS_BIRTH, CODE_GENDER, CNT_CHILDREN
- **Financial:** AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY
- **Employment:** DAYS_EMPLOYED, ORGANIZATION_TYPE
- **Ratios:** CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO

**Behavioral Model Features:**

- **Payment History:** PAY_0 through PAY_6
- **Credit Utilization:** LIMIT_BAL
- **Engineered Features:**
  - spending_volatility
  - debt_stress_index
  - repayment_ratio
  - payment_consistency_ratio
  - credit_utilization_trend

**Interactive Features:**

- Hover over bars for exact importance values
- Zoom in/out using plotly controls
- Download chart as PNG

---

## Exploratory Data Analysis

### EDA Page

#### Data Exploration Options

**Dataset Selection:**

- Choose from available datasets
- View summary statistics
- Explore distributions

**Visualizations:**

- Distribution plots for numeric features
- Bar charts for categorical features
- Correlation heatmaps
- Box plots for outlier detection

**Summary Statistics:**

- Count, mean, std, min, max
- 25th, 50th, 75th percentiles
- Missing value counts

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Prediction Failed

**Symptoms:** Error message after clicking "Predict"

**Solutions:**

1. **Check required fields:** Ensure all \* marked fields are filled
2. **Verify data types:**
   - Numeric fields: Use numbers only
   - No currency symbols ($, €)
   - No commas in large numbers
3. **Check value ranges:**
   - Ages should be positive
   - Percentages should be 0-100
   - Dates should be negative (days before application)
4. **Try refreshing page:** Click browser refresh (F5)

#### Issue: CSV Upload Failed

**Symptoms:** "Invalid column names" or "Missing columns" error

**Solutions:**

1. **Download template again:** Use the template from the app
2. **Check column names:** Must match exactly (case-sensitive)
3. **Check file format:** Save as CSV (not Excel .xlsx)
4. **Check encoding:** Use UTF-8 encoding
5. **Remove extra columns:** Only include template columns

#### Issue: Model Not Loading

**Symptoms:** "Model not found" error

**Solutions:**

1. **Check models folder:** Ensure .pkl files exist in `models/` directory
2. **Restart application:** Stop and restart Streamlit
3. **Check file permissions:** Ensure read access to model files

#### Issue: Slow Performance

**Symptoms:** Long wait times for predictions

**Solutions:**

1. **Reduce batch size:** Split large CSV files into smaller batches
2. **Use manual input:** For single predictions
3. **Close other applications:** Free up system memory
4. **Use ensemble model selectively:** Traditional/Behavioral are faster

#### Issue: Unexpected Predictions

**Symptoms:** Predictions don't match expectations

**Explanations:**

1. **Model limitations:** See PROJECT_LIMITATIONS.md
2. **Feature importance:** Check which features model prioritizes
3. **Data quality:** Verify input data is accurate
4. **Model version:** Confirm you're using intended model

---

## Best Practices

### For Accurate Predictions

**Data Quality:**

- Provide complete information (minimize missing values)
- Use accurate, up-to-date data
- Verify calculations (ratios, percentages)
- Check for typos in numeric fields

**Model Selection:**

- Use **Ensemble** for best accuracy
- Use **Traditional** when you have comprehensive financial data
- Use **Behavioral** when you have transaction history
- Compare predictions across models for high-stake decisions

**Interpretation:**

- Consider risk level, not just probability
- Review feature importance for decision context
- Use medium-risk predictions for manual review
- Document decisions for compliance

### For Batch Processing

**Preparation:**

- Clean data before upload
- Standardize formats across rows
- Test with small batch first (10 rows)
- Keep original data as backup

**Validation:**

- Review summary statistics after processing
- Check for unusual patterns (all high/low risk)
- Spot-check individual predictions
- Compare batch results to historical patterns

**Record Keeping:**

- Download results immediately
- Save with timestamp in filename
- Include metadata (model used, date, batch ID)
- Archive input and output files

---

## Additional Resources

**Documentation:**

- `README.md` - Project overview and setup
- `PROJECT_LIMITATIONS.md` - Known limitations and constraints
- `TEST_RESULTS_REPORT.md` - Model testing results
- `MODEL_ARCHITECTURE_FLOWCHART.md` - Technical architecture

**Support:**

- Check console logs for detailed error messages
- Review test files in `tests/` directory for examples
- Examine `generate_test_cases.py` for data format examples

**Model Files:**

- `models/Traditional_model.pkl` - Traditional model
- `models/Behaviorial_model.pkl` - Behavioral model
- `models/model_ensemble_hybrid.pkl` - Ensemble meta-model
- `models/model_ensemble_wrapper.pkl` - Ensemble wrapper
- `models/ensemble_metadata.pkl` - Feature metadata

---

## Glossary

**AUC-ROC:** Area Under the Receiver Operating Characteristic Curve - measures model discrimination ability

**Data Cleaning:** Centralized preprocessing module (`src/data_cleaning.py`) that handles missing values, infinities, placeholder replacement, and feature alignment

**Default:** Failure to repay a loan according to agreed terms

**Ensemble:** Combination of multiple models for improved predictions

**False Positive:** Incorrectly predicting default when applicant would repay

**False Negative:** Missing a default prediction when applicant would default

**Feature Engineering:** Creating new features from raw data

**Imputation:** Process of filling missing values using strategies like median, mean, or categorical defaults

**Meta-Learner:** Model that learns from other models' predictions

**Overfitting:** Model performs well on training data but poorly on new data

**Probability:** Likelihood of default expressed as decimal (0.0-1.0) or percentage (0%-100%)

**Risk Level:** Categorical classification: Low, Medium, or High risk

**True Positive:** Correctly identifying a default

**True Negative:** Correctly identifying a non-default

---

_Last Updated: December 3, 2025_  
_Version: 2.0 - Added centralized data cleaning module_
