"""
Ensemble Hybrid Model Class - 538-Feature Architecture
Wrapper for combining traditional and behavioral models with meta-feature generation

Architecture:
- Base Models: Traditional (487 features) + Behavioral (44 features)
- Meta-Features: 7 engineered features from base model predictions
- Combined: 7 meta + 487 traditional + 44 behavioral = 538 total features
- Meta-Learner: CatBoost with auto class balancing

Performance:
- Test AUC: 0.8590
- Accuracy: 81%
- Recall: 77%
"""
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', message='.*number of features.*')
warnings.filterwarnings('ignore', message='.*predict_disable_shape_check.*')

class EnsembleHybridModel:
    """
    Wrapper class for 538-feature ensemble prediction
    
    Generates meta-features from base model predictions and combines with raw features
    for comprehensive risk assessment using CatBoost meta-learner.
    """
    def __init__(self, meta_model, model_trad, model_behav, trad_feats, behav_feats):
        self.meta_model = meta_model
        self.model_traditional = model_trad
        self.model_behavioral = model_behav
        self.traditional_features = trad_feats  # Keep ALL features including IDs for base model predictions
        self.behavioral_features = behav_feats   # Keep ALL features including IDs for base model predictions
        
        # For CatBoost ensemble trained with only 7 meta-features
        # No additional key features needed (simplified approach)
        self.key_traditional = []
        self.key_behavioral = []
    
    def predict_proba(self, X):
        """Predict probabilities using the ensemble"""
        # Prepare features for each base model
        X_trad = X[self.traditional_features].copy()
        X_behav = X[self.behavioral_features].copy()
        
        # Handle missing values using centralized data cleaning
        from src.data_cleaning import impute_categorical_columns, impute_numeric_columns
        
        X_trad = impute_categorical_columns(X_trad, fill_value='MISSING')
        X_trad = impute_numeric_columns(X_trad, strategy='median')
        
        # Encode categorical after imputation
        obj_cols_before = (X_trad.dtypes == 'object').sum()
        for col in X_trad.columns:
            if X_trad[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                X_trad[col] = le.fit_transform(X_trad[col].astype(str))
        obj_cols_after = (X_trad.dtypes == 'object').sum()
        
        if obj_cols_before > 0 and obj_cols_after == 0:
            pass  # Successfully encoded
        elif obj_cols_after > 0:
            import warnings
            warnings.warn(f"Warning: {obj_cols_after} object columns remaining in X_trad after encoding")
        
        X_behav = impute_categorical_columns(X_behav, fill_value='MISSING')
        X_behav = impute_numeric_columns(X_behav, strategy='median')
        
        # Encode categorical after imputation
        for col in X_behav.columns:
            if X_behav[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                X_behav[col] = le.fit_transform(X_behav[col].astype(str))
        
        # Get base model predictions (7 meta-features)
        pred_trad = self.model_traditional.predict_proba(X_trad)[:, 1]
        pred_behav = self.model_behavioral.predict_proba(X_behav)[:, 1]
        
        # Create 7 meta-features from predictions
        meta_features = pd.DataFrame({
            'pred_traditional': pred_trad,
            'pred_behavioral': pred_behav,
            'pred_avg': (pred_trad + pred_behav) / 2,
            'pred_max': np.maximum(pred_trad, pred_behav),
            'pred_min': np.minimum(pred_trad, pred_behav),
            'pred_diff': np.abs(pred_trad - pred_behav),
            'pred_ratio': pred_trad / (pred_behav + 0.001)
        })
        
        # Combine: 7 meta-features + 531 raw features = 538 total features
        meta_X = pd.concat([meta_features, X_trad, X_behav], axis=1)
        
        # Get final prediction from meta-model
        # Handle both LightGBM (best_iteration) and CatBoost (best_iteration_)
        if hasattr(self.meta_model, 'best_iteration_'):
            # CatBoost - use predict_proba for probabilities
            final_proba = self.meta_model.predict_proba(meta_X)[:, 1]
        elif hasattr(self.meta_model, 'best_iteration'):
            # LightGBM
            final_proba = self.meta_model.predict(meta_X, num_iteration=self.meta_model.best_iteration)
        else:
            # No best iteration attribute, use default
            try:
                final_proba = self.meta_model.predict_proba(meta_X)[:, 1]
            except:
                final_proba = self.meta_model.predict(meta_X)
        
        return np.column_stack([1 - final_proba, final_proba])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
