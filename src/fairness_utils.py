"""
Fairness Utilities - Helper functions for fairness analysis

This module provides utility functions for calculating and evaluating
fairness metrics across protected demographic attributes.
"""

import numpy as np
import pandas as pd


def calculate_fairness_metrics(y_true, y_pred, sensitive_feature, feature_name):
    """
    Calculate comprehensive fairness metrics for a protected attribute
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    sensitive_feature : array
        Protected attribute values (e.g., gender, age group)
    feature_name : str
        Name of the protected attribute
        
    Returns:
    --------
    dict : Dictionary containing fairness metrics including:
        - groups: Per-group metrics (acceptance rate, TPR, FPR, sample size)
        - overall: Overall fairness metrics (disparate impact ratio, passes 80% rule, demographic parity diff)
    """
    
    results = {
        'feature': feature_name,
        'groups': {},
        'overall': {}
    }
    
    unique_groups = np.unique(sensitive_feature)
    acceptance_rates = []
    disparate_impacts = []
    
    # Calculate per-group metrics
    for group in unique_groups:
        mask = sensitive_feature == group
        group_pred = y_pred[mask]
        group_true = y_true[mask]
        
        # Basic metrics
        acceptance_rate = group_pred.mean()
        n_samples = len(group_pred)
        
        # TPR and FPR
        if group_true.sum() > 0:  # Has positive samples
            tpr = group_pred[group_true == 1].mean()
        else:
            tpr = None
            
        if (group_true == 0).sum() > 0:  # Has negative samples
            fpr = group_pred[group_true == 0].mean()
        else:
            fpr = None
        
        results['groups'][group] = {
            'acceptance_rate': acceptance_rate,
            'n_samples': n_samples,
            'tpr': tpr,
            'fpr': fpr,
            'n_defaults': int(group_true.sum())
        }
        
        acceptance_rates.append(acceptance_rate)
    
    # Calculate disparate impact (80% rule)
    if len(acceptance_rates) > 0:
        max_rate = max(acceptance_rates)
        min_rate = min(acceptance_rates)
        
        if max_rate > 0:
            disparate_impact_ratio = min_rate / max_rate
        else:
            disparate_impact_ratio = 0.0
        
        results['overall']['disparate_impact_ratio'] = disparate_impact_ratio
        results['overall']['passes_80_rule'] = disparate_impact_ratio >= 0.8
        results['overall']['demographic_parity_diff'] = max_rate - min_rate
    
    return results


def extract_protected_attributes(data):
    """
    Extract protected attributes from data for fairness evaluation
    
    Parameters:
    -----------
    data : DataFrame
        Input data containing protected attributes
        
    Returns:
    --------
    dict : Dictionary mapping attribute names to their values
    """
    protected_attrs = {}
    
    # Look for protected attributes with various prefixes
    attr_names = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE_GROUP']
    prefixes = ['behav_', 'trad_', '']
    
    for attr in attr_names:
        for prefix in prefixes:
            col_name = f"{prefix}{attr}"
            if col_name in data.columns:
                protected_attrs[attr] = data[col_name].values
                break
    
    # Create AGE_GROUP from DAYS_BIRTH if not found
    if 'AGE_GROUP' not in protected_attrs:
        for col in data.columns:
            if 'DAYS_BIRTH' in col:
                age_years = -data[col] / 365.25
                age_group = pd.cut(age_years, 
                                  bins=[0, 30, 40, 50, 60, 100], 
                                  labels=[0, 1, 2, 3, 4],
                                  right=False)
                protected_attrs['AGE_GROUP'] = age_group.astype(int).values
                break
    
    return protected_attrs
