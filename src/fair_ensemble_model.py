"""
Fair Ensemble Model - Fairness-aware wrapper for ensemble predictions

This module provides classes for fairness-aware machine learning predictions
using threshold optimization to achieve demographic parity.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for the ensemble model.
    
    This wrapper allows the ensemble model to be used with fairlearn's
    ThresholdOptimizer which requires sklearn-compatible estimators.
    """
    
    def __init__(self, model=None):
        """
        Initialize the wrapper
        
        Parameters:
        -----------
        model : EnsembleHybridModel
            The trained ensemble model to wrap
        """
        self.model = model
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Required by sklearn's BaseEstimator for compatibility with
        ThresholdOptimizer and other sklearn utilities.
        """
        return {"model": self.model}
    
    def set_params(self, **parameters):
        """
        Set parameters for this estimator.
        
        Required by sklearn's BaseEstimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        """
        Fit method (no-op since model is already trained)
        
        The ensemble model is pre-trained, so this method just returns self
        to maintain sklearn compatibility.
        """
        return self
    
    def predict(self, X):
        """
        Make binary predictions
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
            
        Returns:
        --------
        array : Binary predictions (0 or 1)
        """
        proba = self.model.predict_proba(X)
        if proba.ndim > 1 and proba.shape[1] >= 2:
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return (proba >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
            
        Returns:
        --------
        array : Probability estimates for each class
        """
        proba = self.model.predict_proba(X)
        
        # Ensure 2D output with probabilities for both classes
        if proba.ndim == 1 or proba.shape[1] == 1:
            # Convert single column to two-class probabilities
            proba_class_1 = proba.ravel() if proba.ndim > 1 else proba
            proba_class_0 = 1 - proba_class_1
            return np.column_stack([proba_class_0, proba_class_1])
        
        return proba


class FairEnsembleModel:
    """
    Wrapper for ensemble model with fairness-aware predictions using threshold optimization.
    
    Uses different decision thresholds for different demographic groups to ensure
    fairness while maintaining predictive performance.
    
    Attributes:
    -----------
    base_model : EnsembleWrapper
        The wrapped base ensemble model
    fair_models : dict
        Dictionary mapping attribute names (e.g., 'SEX', 'MARRIAGE') to 
        ThresholdOptimizer models for each protected attribute
    """
    
    def __init__(self, base_model, fair_models_dict):
        """
        Initialize Fair Ensemble Model
        
        Parameters:
        -----------
        base_model : EnsembleWrapper
            The wrapped base ensemble model
        fair_models_dict : dict
            Dictionary mapping attribute names to ThresholdOptimizer models
        """
        self.base_model = base_model
        self.fair_models = fair_models_dict
        
    def predict_fair(self, X, protected_attrs_dict, use_fairness=True):
        """
        Make fair predictions using threshold optimization
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        protected_attrs_dict : dict
            Dictionary of protected attributes {'SEX': array, 'MARRIAGE': array, 'AGE_GROUP': array}
        use_fairness : bool
            Whether to use fair predictions (True) or baseline (False)
            
        Returns:
        --------
        dict or array : Fair predictions for each attribute or baseline predictions
        """
        if not use_fairness:
            # Return baseline predictions
            return self.base_model.predict(X)
        
        # Get fair predictions for each attribute
        fair_predictions = {}
        
        for attr_name, sensitive_feature in protected_attrs_dict.items():
            if attr_name in self.fair_models:
                fair_pred = self.fair_models[attr_name].predict(
                    X, 
                    sensitive_features=sensitive_feature
                )
                fair_predictions[attr_name] = fair_pred
        
        return fair_predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities from base model
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
            
        Returns:
        --------
        array : Probability estimates
        """
        return self.base_model.predict_proba(X)
    
    def evaluate_fairness(self, X, y_true, protected_attrs_dict):
        """
        Evaluate fairness metrics for all protected attributes
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y_true : array
            True labels
        protected_attrs_dict : dict
            Dictionary of protected attributes
            
        Returns:
        --------
        dict : Fairness metrics for both baseline and fair predictions
        """
        from .fairness_utils import calculate_fairness_metrics
        
        results = {
            'baseline': {},
            'fair': {}
        }
        
        # Baseline predictions
        y_pred_baseline = self.base_model.predict(X)
        
        for attr_name, sensitive_feature in protected_attrs_dict.items():
            # Baseline fairness
            baseline_metrics = calculate_fairness_metrics(
                y_true, y_pred_baseline, sensitive_feature, attr_name
            )
            results['baseline'][attr_name] = baseline_metrics
            
            # Fair predictions (if available)
            if attr_name in self.fair_models:
                y_pred_fair = self.fair_models[attr_name].predict(
                    X, sensitive_features=sensitive_feature
                )
                fair_metrics = calculate_fairness_metrics(
                    y_true, y_pred_fair, sensitive_feature, attr_name
                )
                results['fair'][attr_name] = fair_metrics
        
        return results
