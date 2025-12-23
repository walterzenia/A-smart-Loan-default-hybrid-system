"""
Fairness Analysis for Loan Default Prediction Models

This module evaluates fairness metrics for Traditional, Behavioral, and Ensemble models
to ensure compliance with fair lending regulations and identify potential biases.

Protected Attributes Analyzed:
- SEX: Gender (1=Male, 2=Female)
- EDUCATION: Education level (1=Graduate, 2=University, 3=High School, 4=Others)
- MARRIAGE: Marital status (1=Married, 2=Single, 3=Others)
- AGE: Age groups (<30, 30-40, 40-50, 50+)

Fairness Metrics:
- Demographic Parity: Equal acceptance rates across groups
- Equalized Odds: Equal TPR and FPR across groups
- Disparate Impact: 80% rule compliance
- Predictive Parity: Equal precision across groups
- Calibration: Prediction accuracy within groups
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend before importing pyplot to avoid Tcl/Tk issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve


class FairnessAnalyzer:
    """
    Comprehensive fairness evaluation for loan default prediction models.
    """
    
    def __init__(self, model_name, model, X_test, y_test):
        """
        Initialize fairness analyzer.
        
        Args:
            model_name: Name of the model being analyzed
            model: Trained model object
            X_test: Test features
            y_test: True labels
        """
        self.model_name = model_name
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            self.y_pred_proba = model.predict(X_test)
        
        self.y_pred = (self.y_pred_proba >= 0.5).astype(int)
        
        # Store results
        self.fairness_results = {}
    
    def demographic_parity(self, protected_attr, attr_name):
        """
        Calculate demographic parity (statistical parity).
        Measures if acceptance rates are similar across groups.
        
        Args:
            protected_attr: Array of protected attribute values
            attr_name: Name of the protected attribute
            
        Returns:
            dict: Fairness metrics for each group
        """
        groups = np.unique(protected_attr)
        results = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            
            acceptance_rate = self.y_pred[mask].mean()
            results[group] = {
                'acceptance_rate': acceptance_rate,
                'n_samples': mask.sum()
            }
        
        # Calculate disparities
        acceptance_rates = [r['acceptance_rate'] for r in results.values()]
        max_disparity = max(acceptance_rates) - min(acceptance_rates)
        
        results['_summary'] = {
            'max_disparity': max_disparity,
            'passes_5pct_threshold': max_disparity <= 0.05
        }
        
        self.fairness_results[f'demographic_parity_{attr_name}'] = results
        return results
    
    def equalized_odds(self, protected_attr, attr_name):
        """
        Calculate equalized odds (equal opportunity + equal FPR).
        Measures if TPR and FPR are similar across groups.
        
        Args:
            protected_attr: Array of protected attribute values
            attr_name: Name of the protected attribute
            
        Returns:
            dict: TPR and FPR for each group
        """
        groups = np.unique(protected_attr)
        results = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            
            y_true_group = self.y_test[mask]
            y_pred_group = self.y_pred[mask]
            
            # Check if we have both classes in the ground truth
            unique_true = np.unique(y_true_group)
            if len(unique_true) < 2:
                # Only one class present, can't compute meaningful TPR/FPR
                results[group] = {
                    'tpr': None,
                    'fpr': None,
                    'n_samples': mask.sum(),
                    'n_positive': (y_true_group == 1).sum(),
                    'n_negative': (y_true_group == 0).sum(),
                    'note': 'Insufficient class diversity for TPR/FPR calculation'
                }
                continue
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true_group, y_pred_group)
            
            # Handle different matrix shapes
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # Only one class predicted
                if unique_true[0] == 0:
                    tn = cm[0, 0]
                    fp, fn, tp = 0, 0, 0
                else:
                    tp = cm[0, 0]
                    tn, fp, fn = 0, 0, 0
            else:
                # Edge case
                results[group] = {
                    'tpr': None,
                    'fpr': None,
                    'n_samples': mask.sum(),
                    'n_positive': (y_true_group == 1).sum(),
                    'n_negative': (y_true_group == 0).sum(),
                    'note': 'Unexpected confusion matrix shape'
                }
                continue
            
            # Calculate TPR and FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results[group] = {
                'tpr': tpr,
                'fpr': fpr,
                'n_samples': mask.sum(),
                'n_positive': (y_true_group == 1).sum(),
                'n_negative': (y_true_group == 0).sum()
            }
        
        # Calculate disparities
        tprs = [r['tpr'] for r in results.values() if isinstance(r, dict) and r.get('tpr') is not None]
        fprs = [r['fpr'] for r in results.values() if isinstance(r, dict) and r.get('fpr') is not None]
        
        tpr_disparity = max(tprs) - min(tprs) if len(tprs) > 1 else 0
        fpr_disparity = max(fprs) - min(fprs) if len(fprs) > 1 else 0
        
        results['_summary'] = {
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'passes_5pct_threshold': (tpr_disparity <= 0.05 and fpr_disparity <= 0.05)
        }
        
        self.fairness_results[f'equalized_odds_{attr_name}'] = results
        return results
    
    def disparate_impact(self, protected_attr, attr_name, reference_group=None):
        """
        Calculate disparate impact ratio (80% rule).
        Ratio of acceptance rates between groups should be >= 0.8.
        
        Args:
            protected_attr: Array of protected attribute values
            attr_name: Name of the protected attribute
            reference_group: Reference group for comparison (default: majority group)
            
        Returns:
            dict: Disparate impact ratios
        """
        groups = np.unique(protected_attr)
        acceptance_rates = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            acceptance_rates[group] = self.y_pred[mask].mean()
        
        # Determine reference group (highest acceptance rate if not specified)
        if reference_group is None or reference_group not in acceptance_rates:
            reference_group = max(acceptance_rates, key=acceptance_rates.get)
        
        reference_rate = acceptance_rates[reference_group]
        
        results = {
            'reference_group': reference_group,
            'reference_rate': reference_rate
        }
        
        # Calculate disparate impact ratios
        for group, rate in acceptance_rates.items():
            if group == reference_group:
                results[group] = {
                    'acceptance_rate': rate,
                    'disparate_impact_ratio': 1.0,
                    'passes_80pct_rule': True
                }
            else:
                di_ratio = rate / reference_rate if reference_rate > 0 else 0
                results[group] = {
                    'acceptance_rate': rate,
                    'disparate_impact_ratio': di_ratio,
                    'passes_80pct_rule': di_ratio >= 0.8
                }
        
        # Overall compliance
        all_ratios = [r['disparate_impact_ratio'] for k, r in results.items() 
                     if isinstance(r, dict) and 'disparate_impact_ratio' in r]
        results['_summary'] = {
            'all_groups_pass': all(r >= 0.8 for r in all_ratios),
            'min_ratio': min(all_ratios) if all_ratios else 0
        }
        
        self.fairness_results[f'disparate_impact_{attr_name}'] = results
        return results
    
    def predictive_parity(self, protected_attr, attr_name):
        """
        Calculate predictive parity (equal precision across groups).
        
        Args:
            protected_attr: Array of protected attribute values
            attr_name: Name of the protected attribute
            
        Returns:
            dict: Precision for each group
        """
        groups = np.unique(protected_attr)
        results = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            
            y_true_group = self.y_test[mask]
            y_pred_group = self.y_pred[mask]
            
            # Calculate precision
            tp = ((y_pred_group == 1) & (y_true_group == 1)).sum()
            fp = ((y_pred_group == 1) & (y_true_group == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            results[group] = {
                'precision': precision,
                'n_predicted_positive': (y_pred_group == 1).sum(),
                'n_samples': mask.sum()
            }
        
        # Calculate disparity
        precisions = [r['precision'] for r in results.values()]
        precision_disparity = max(precisions) - min(precisions) if len(precisions) > 0 else 0
        
        results['_summary'] = {
            'precision_disparity': precision_disparity,
            'passes_5pct_threshold': precision_disparity <= 0.05
        }
        
        self.fairness_results[f'predictive_parity_{attr_name}'] = results
        return results
    
    def calibration_by_group(self, protected_attr, attr_name, n_bins=10):
        """
        Evaluate calibration within each demographic group.
        
        Args:
            protected_attr: Array of protected attribute values
            attr_name: Name of the protected attribute
            n_bins: Number of bins for calibration curve
            
        Returns:
            dict: Calibration metrics for each group
        """
        groups = np.unique(protected_attr)
        results = {}
        
        for group in groups:
            mask = protected_attr == group
            if mask.sum() < n_bins:  # Need enough samples
                continue
            
            y_true_group = self.y_test[mask]
            y_pred_proba_group = self.y_pred_proba[mask]
            
            try:
                # Calculate calibration curve
                fraction_positives, mean_predicted = calibration_curve(
                    y_true_group, y_pred_proba_group, n_bins=n_bins, strategy='uniform'
                )
                
                # Calculate expected calibration error (ECE)
                ece = np.mean(np.abs(fraction_positives - mean_predicted))
                
                results[group] = {
                    'expected_calibration_error': ece,
                    'n_samples': mask.sum(),
                    'fraction_positives': fraction_positives.tolist(),
                    'mean_predicted': mean_predicted.tolist()
                }
            except:
                results[group] = {
                    'expected_calibration_error': None,
                    'n_samples': mask.sum(),
                    'error': 'Insufficient data for calibration'
                }
        
        # Summary
        eces = [r['expected_calibration_error'] for r in results.values() 
               if r.get('expected_calibration_error') is not None]
        
        results['_summary'] = {
            'max_ece': max(eces) if eces else None,
            'mean_ece': np.mean(eces) if eces else None
        }
        
        self.fairness_results[f'calibration_{attr_name}'] = results
        return results
    
    def analyze_all(self, protected_attrs):
        """
        Run all fairness analyses for given protected attributes.
        
        Args:
            protected_attrs: Dictionary mapping attribute names to arrays
            
        Returns:
            dict: All fairness results
        """
        print(f"\n{'='*60}")
        print(f"Fairness Analysis: {self.model_name}")
        print(f"{'='*60}\n")
        
        for attr_name, attr_values in protected_attrs.items():
            print(f"\nAnalyzing protected attribute: {attr_name}")
            print(f"-" * 50)
            
            # Run all fairness tests
            dp = self.demographic_parity(attr_values, attr_name)
            print(f"  [OK] Demographic Parity: {'PASS' if dp['_summary']['passes_5pct_threshold'] else 'FAIL'}")
            
            eo = self.equalized_odds(attr_values, attr_name)
            print(f"  [OK] Equalized Odds: {'PASS' if eo['_summary']['passes_5pct_threshold'] else 'FAIL'}")
            
            di = self.disparate_impact(attr_values, attr_name)
            print(f"  [OK] Disparate Impact (80% rule): {'PASS' if di['_summary']['all_groups_pass'] else 'FAIL'}")
            
            pp = self.predictive_parity(attr_values, attr_name)
            print(f"  [OK] Predictive Parity: {'PASS' if pp['_summary']['passes_5pct_threshold'] else 'FAIL'}")
            
            cal = self.calibration_by_group(attr_values, attr_name)
            print(f"  [OK] Calibration analysis complete")
        
        return self.fairness_results
    
    def generate_report(self, output_file=None):
        """
        Generate comprehensive fairness report.
        
        Args:
            output_file: Path to save report (optional)
        """
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"FAIRNESS ANALYSIS REPORT: {self.model_name}")
        report.append(f"{'='*80}\n")
        
        for metric_key, metric_results in self.fairness_results.items():
            report.append(f"\n{metric_key.upper()}")
            report.append("-" * 80)
            
            for group, values in metric_results.items():
                if group == '_summary':
                    report.append(f"\n  Summary:")
                    for k, v in values.items():
                        report.append(f"    {k}: {v}")
                else:
                    report.append(f"\n  Group {group}:")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            if not isinstance(v, list):
                                report.append(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")
        
        return report_text
    
    def plot_fairness_metrics(self, output_dir=None):
        """
        Create visualizations of fairness metrics.
        
        Args:
            output_dir: Directory to save plots (optional)
        """
        import os
        
        # Use non-interactive backend for matplotlib to avoid Tcl/Tk issues
        import matplotlib
        matplotlib.use('Agg')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Acceptance rates by protected attribute
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Fairness Metrics: {self.model_name}', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for metric_key in self.fairness_results:
            if 'demographic_parity' in metric_key:
                attr_name = metric_key.split('_')[-1]
                results = self.fairness_results[metric_key]
                
                groups = [k for k in results.keys() if k != '_summary']
                rates = [results[g]['acceptance_rate'] for g in groups]
                
                ax = axes[plot_idx // 2, plot_idx % 2]
                bars = ax.bar(range(len(groups)), rates, color='steelblue', alpha=0.7)
                ax.axhline(y=np.mean(rates), color='red', linestyle='--', label='Mean', linewidth=2)
                ax.set_xlabel(attr_name, fontsize=12)
                ax.set_ylabel('Acceptance Rate', fontsize=12)
                ax.set_title(f'Acceptance Rate by {attr_name}', fontsize=13, fontweight='bold')
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(groups)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/acceptance_rates_{self.model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Disparate Impact Ratios
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Disparate Impact Analysis: {self.model_name}', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for metric_key in self.fairness_results:
            if 'disparate_impact' in metric_key:
                attr_name = metric_key.split('_')[-1]
                results = self.fairness_results[metric_key]
                
                groups = [k for k in results.keys() if k != '_summary' and k != 'reference_group' and k != 'reference_rate']
                ratios = [results[g]['disparate_impact_ratio'] for g in groups]
                colors = ['green' if r >= 0.8 else 'red' for r in ratios]
                
                ax = axes[plot_idx // 2, plot_idx % 2]
                bars = ax.bar(range(len(groups)), ratios, color=colors, alpha=0.7)
                ax.axhline(y=0.8, color='orange', linestyle='--', label='80% Rule', linewidth=2)
                ax.axhline(y=1.0, color='blue', linestyle='--', label='Parity', linewidth=2)
                ax.set_xlabel(attr_name, fontsize=12)
                ax.set_ylabel('Disparate Impact Ratio', fontsize=12)
                ax.set_title(f'Disparate Impact by {attr_name}', fontsize=13, fontweight='bold')
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(groups)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, max(ratios) * 1.2 if ratios else 1.5)
                
                plot_idx += 1
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/disparate_impact_{self.model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()


def prepare_behavioral_data(test_file):
    """
    Load and prepare behavioral model test data with protected attributes.
    
    Args:
        test_file: Path to behavioral model test CSV
        
    Returns:
        X_test, y_test, protected_attrs
    """
    df = pd.read_csv(test_file)
    
    # Identify target column
    if 'TARGET' in df.columns:
        y_test = df['TARGET'].values
        X_test = df.drop('TARGET', axis=1)
    elif 'default.payment.next.month' in df.columns:
        y_test = df['default.payment.next.month'].values
        X_test = df.drop('default.payment.next.month', axis=1)
    else:
        raise ValueError("Target column not found")
    
    # Extract protected attributes
    protected_attrs = {}
    
    if 'SEX' in X_test.columns:
        protected_attrs['SEX'] = X_test['SEX'].values
    
    if 'EDUCATION' in X_test.columns:
        protected_attrs['EDUCATION'] = X_test['EDUCATION'].values
    
    if 'MARRIAGE' in X_test.columns:
        protected_attrs['MARRIAGE'] = X_test['MARRIAGE'].values
    
    if 'AGE' in X_test.columns:
        # Bin age into groups
        age_bins = [0, 30, 40, 50, 100]
        age_labels = ['<30', '30-40', '40-50', '50+']
        protected_attrs['AGE_GROUP'] = pd.cut(X_test['AGE'], bins=age_bins, labels=age_labels).values
    
    return X_test, y_test, protected_attrs

def prepare_traditional_data(test_file):
    """
    Load and prepare traditional model test data with protected attributes.
    
    Args:
        test_file: Path to traditional model test CSV
        
    Returns:
        X_test, y_test, protected_attrs
    """
    from sklearn.preprocessing import LabelEncoder
    from src.data_cleaning import impute_categorical_columns, impute_numeric_columns
    
    df = pd.read_csv(test_file)
    
    # Identify target column
    if 'TARGET' in df.columns:
        y_test = df['TARGET'].values
        X_test = df.drop('TARGET', axis=1)
    else:
        raise ValueError("TARGET column not found in traditional test data")
    
    # Extract protected attributes BEFORE encoding (for fairness analysis)
    protected_attrs = {}
    
    # Gender: CODE_GENDER (F=Female, M=Male, XNA=Unknown)
    if 'CODE_GENDER' in X_test.columns:
        # Map to numeric for consistency with other models
        gender_map = {'M': 1.0, 'F': 2.0, 'XNA': 0.0}
        protected_attrs['SEX'] = X_test['CODE_GENDER'].map(gender_map).fillna(0.0).values
    
    # Education: NAME_EDUCATION_TYPE
    if 'NAME_EDUCATION_TYPE' in X_test.columns:
        # Map education levels to numeric groups
        edu_map = {
            'Lower secondary': 1,
            'Secondary / secondary special': 2,
            'Incomplete higher': 3,
            'Higher education': 4,
            'Academic degree': 5
        }
        protected_attrs['EDUCATION'] = X_test['NAME_EDUCATION_TYPE'].map(edu_map).fillna(2).values
    
    # Marital Status: NAME_FAMILY_STATUS
    if 'NAME_FAMILY_STATUS' in X_test.columns:
        # Map family status to numeric groups
        family_map = {
            'Married': 1,
            'Civil marriage': 1,  # Group with married
            'Single / not married': 2,
            'Separated': 3,
            'Widow': 3  # Group with separated
        }
        protected_attrs['MARRIAGE'] = X_test['NAME_FAMILY_STATUS'].map(family_map).fillna(2).values
    
    # Age: DAYS_BIRTH (negative days since birth)
    if 'DAYS_BIRTH' in X_test.columns:
        # Convert days to years and bin into age groups
        age_years = -X_test['DAYS_BIRTH'] / 365.25
        age_bins = [0, 30, 40, 50, 100]
        age_labels = ['<30', '30-40', '40-50', '50+']
        protected_attrs['AGE_GROUP'] = pd.cut(age_years, bins=age_bins, labels=age_labels).values
    
    # Now clean and encode X_test for model prediction
    X_test = impute_categorical_columns(X_test, fill_value='MISSING')
    X_test = impute_numeric_columns(X_test, strategy='median')
    
    # Encode ALL categorical columns (same approach as ensemble model)
    for col in X_test.columns:
        if X_test[col].dtype in ['object', 'category']:
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
    
    return X_test, y_test, protected_attrs

def prepare_ensemble_data(test_file):
    """
    Load and prepare ensemble model test data with protected attributes.
    
    Args:
        test_file: Path to ensemble test CSV
        
    Returns:
        X_test, y_test, protected_attrs
    """
    df = pd.read_csv(test_file)
    
    # Identify target column
    if 'TARGET' in df.columns:
        y_test = df['TARGET'].values
        X_test = df.drop('TARGET', axis=1)
    else:
        raise ValueError("Target column not found")
    
    # Extract protected attributes - check both with and without behav_ prefix
    protected_attrs = {}
    
    # SEX
    if 'behav_SEX' in X_test.columns:
        protected_attrs['SEX'] = X_test['behav_SEX'].values
    elif 'SEX' in X_test.columns:
        protected_attrs['SEX'] = X_test['SEX'].values
    
    # EDUCATION
    if 'behav_EDUCATION' in X_test.columns:
        protected_attrs['EDUCATION'] = X_test['behav_EDUCATION'].values
    elif 'EDUCATION' in X_test.columns:
        protected_attrs['EDUCATION'] = X_test['EDUCATION'].values
    
    # MARRIAGE
    if 'behav_MARRIAGE' in X_test.columns:
        protected_attrs['MARRIAGE'] = X_test['behav_MARRIAGE'].values
    elif 'MARRIAGE' in X_test.columns:
        protected_attrs['MARRIAGE'] = X_test['MARRIAGE'].values
    
    # AGE
    age_col = None
    if 'behav_AGE' in X_test.columns:
        age_col = 'behav_AGE'
    elif 'AGE' in X_test.columns:
        age_col = 'AGE'
    
    if age_col is not None:
        # Bin age into groups
        age_bins = [0, 30, 40, 50, 100]
        age_labels = ['<30', '30-40', '40-50', '50+']
        protected_attrs['AGE_GROUP'] = pd.cut(X_test[age_col], bins=age_bins, labels=age_labels).values
    
    return X_test, y_test, protected_attrs


def main():
    """
    Main function to run fairness analysis on all models.
    """
    import os
    import sys
    
    # Add project root to path for imports
    base_dir = r"c:\Users\user\Desktop\Loan Default Hybrid System"
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    # Paths
    models_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "fairness_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE FAIRNESS ANALYSIS")
    print("Loan Default Prediction Hybrid System")
    print("="*80)
    
    # ========== BEHAVIORAL MODEL ==========
    print("\n\n[1/4] Analyzing Behavioral Model...")
    print("-" * 80)
    
    try:
        # Load behavioral model using joblib (safer for LightGBM models)
        import joblib
        behavioral_model = joblib.load(os.path.join(models_dir, "Behaviorial_model.pkl"))
        
        # Load test data
        X_test_behav, y_test_behav, protected_attrs_behav = prepare_behavioral_data(
            os.path.join(models_dir, "Behaviorial_model_test_data.csv")
        )
        
        # Run fairness analysis
        analyzer_behav = FairnessAnalyzer("Behavioral Model", behavioral_model, X_test_behav, y_test_behav)
        results_behav = analyzer_behav.analyze_all(protected_attrs_behav)
        
        # Generate report
        analyzer_behav.generate_report(os.path.join(output_dir, "behavioral_model_fairness_report.txt"))
        
        # Create visualizations
        analyzer_behav.plot_fairness_metrics(output_dir)
        
        print("\n[OK] Behavioral model analysis complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Error analyzing behavioral model: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========== TRADITIONAL MODEL ==========
    print("\n\n[2/4] Analyzing Traditional Model...")
    print("-" * 80)
    
    try:
        # Load traditional model using joblib
        import joblib
        traditional_model = joblib.load(os.path.join(models_dir, "Traditional_model.pkl"))
        
        # Load test data
        X_test_trad, y_test_trad, protected_attrs_trad = prepare_traditional_data(
            os.path.join(models_dir, "Traditional_model_test_data.csv")
        )
        
        # Run fairness analysis
        analyzer_trad = FairnessAnalyzer("Traditional Model", traditional_model, X_test_trad, y_test_trad)
        results_trad = analyzer_trad.analyze_all(protected_attrs_trad)
        
        # Generate report
        analyzer_trad.generate_report(os.path.join(output_dir, "traditional_model_fairness_report.txt"))
        
        # Create visualizations
        analyzer_trad.plot_fairness_metrics(output_dir)
        
        print("\n[OK] Traditional model analysis complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Error analyzing traditional model: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========== ENSEMBLE MODEL ==========
    print("\n\n[3/4] Analyzing Ensemble Model...")
    print("-" * 80)
    
    try:
        # Load ensemble model using joblib
        import joblib
        ensemble_model = joblib.load(os.path.join(models_dir, "model_ensemble_wrapper.pkl"))
        
        # Load test data
        X_test_ens, y_test_ens, protected_attrs_ens = prepare_ensemble_data(
            os.path.join(base_dir, "data", "test_ensemble_hybrid_preprocessed.csv")
        )
        
        # Run fairness analysis
        analyzer_ens = FairnessAnalyzer("Ensemble Model", ensemble_model, X_test_ens, y_test_ens)
        results_ens = analyzer_ens.analyze_all(protected_attrs_ens)
        
        # Generate report
        analyzer_ens.generate_report(os.path.join(output_dir, "ensemble_model_fairness_report.txt"))
        
        # Create visualizations
        analyzer_ens.plot_fairness_metrics(output_dir)
        
        print("\n[OK] Ensemble model analysis complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Error analyzing ensemble model: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========== SUMMARY ==========
    print("\n\n" + "="*80)
    print("FAIRNESS ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nReports saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - behavioral_model_fairness_report.txt")
    print("  - ensemble_model_fairness_report.txt")
    print("  - acceptance_rates_*.png")
    print("  - disparate_impact_*.png")
    print("\nNext Steps:")
    print("  1. Review fairness reports for any violations")
    print("  2. Check if disparate impact ratios meet 80% rule")
    print("  3. Examine demographic parity and equalized odds")
    print("  4. If biases detected, consider fairness-aware training or post-processing")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
