"""
Train an ensemble hybrid model combining:
1. Traditional_model.pkl (traditional Home Credit features)
2. Behaviorial_model.pkl (behavioral UCI Credit Card features)

This creates a true hybrid model that leverages both feature sets.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import lightgbm as lgb

# Import the ensemble model class from the module
import sys
sys.path.append(str(Path(__file__).parent))
from ensemble_model import EnsembleHybridModel

def load_models():
    """Load both pre-trained models"""
    models_dir = Path(__file__).parent.parent / "models"
    
    model_traditional = joblib.load(models_dir / "Traditional_model.pkl")
    model_behavioral = joblib.load(models_dir / "Behaviorial_model.pkl")
    
    print("✓ Loaded Traditional_model.pkl (traditional features)")
    print("✓ Loaded Behaviorial_model.pkl (behavioral features)")
    
    return model_traditional, model_behavioral

def load_hybrid_data():
    """Load the hybrid dataset with both feature types"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load smoke_engineered (has TARGET)
    smoke_df = pd.read_csv(data_dir / "smoke_engineered.csv")
    
    # Load the hybrid features we created
    smoke_hybrid = pd.read_csv(data_dir / "smoke_hybrid_features.csv")
    
    print(f"\nLoaded hybrid dataset: {smoke_hybrid.shape}")
    
    return smoke_hybrid

def prepare_feature_sets(df, model_traditional, model_behavioral):
    """
    Separate features for each model based on their expected inputs.
    Keep all features the models were trained with (including IDs for prediction),
    but we'll exclude IDs when selecting key features for meta-learning.
    """
    # Get feature names each model expects (keep ALL features they were trained with)
    if hasattr(model_traditional, 'feature_names_in_'):
        traditional_features = list(model_traditional.feature_names_in_)
    elif hasattr(model_traditional, 'feature_name_'):
        traditional_features = list(model_traditional.feature_name_)
    else:
        # Fallback: assume original smoke columns
        traditional_features = [col for col in df.columns if not col.startswith('BILL_') and 
                               not col.startswith('PAY_') and col not in ['TARGET']]
    
    if hasattr(model_behavioral, 'feature_names_in_'):
        behavioral_features = list(model_behavioral.feature_names_in_)
    elif hasattr(model_behavioral, 'feature_name_'):
        behavioral_features = list(model_behavioral.feature_name_)
    else:
        # Behavioral features we created
        behavioral_features = [col for col in df.columns if any(x in col for x in [
            'total_billed', 'total_payment', 'avg_transaction', 'max_billed', 'max_payment',
            'spending_volatility', 'income_consistency', 'rolling_balance', 'net_flow',
            'debt_stress', 'repayment_ratio', 'payment_consistency', 'spend_to_income',
            'max_to_mean', 'missed_payment', 'credit_utilization_trend'
        ])]
    
    # Add missing features to dataframe (filled with 0) to match what models expect
    missing_trad = [f for f in traditional_features if f not in df.columns]
    missing_behav = [f for f in behavioral_features if f not in df.columns]
    
    if missing_trad:
        print(f"\n⚠ Adding {len(missing_trad)} missing traditional features (filled with 0)")
        for feat in missing_trad:
            df[feat] = 0
            print(f"  - {feat}")
    
    if missing_behav:
        print(f"\n⚠ Adding {len(missing_behav)} missing behavioral features (filled with 0)")
        for feat in missing_behav:
            df[feat] = 0
            print(f"  - {feat}")
    
    print(f"\nTraditional features: {len(traditional_features)}")
    print(f"Behavioral features: {len(behavioral_features)}")
    
    return traditional_features, behavioral_features

def create_meta_features(df, model_traditional, model_behavioral, 
                         traditional_features, behavioral_features):
    """
    Create meta-features by getting predictions from both base models
    """
    print("\nGenerating meta-features from base models...")
    
    # Prepare feature sets (features already added in prepare_feature_sets)
    X_traditional = df[traditional_features].copy()
    X_behavioral = df[behavioral_features].copy()
    
    # Fill missing values - handle numeric and categorical separately
    for col in X_traditional.columns:
        if X_traditional[col].dtype in ['object', 'category']:
            X_traditional[col] = X_traditional[col].fillna('MISSING')
        else:
            X_traditional[col] = X_traditional[col].fillna(X_traditional[col].median())
    
    for col in X_behavioral.columns:
        if X_behavioral[col].dtype in ['object', 'category']:
            X_behavioral[col] = X_behavioral[col].fillna('MISSING')
        else:
            X_behavioral[col] = X_behavioral[col].fillna(X_behavioral[col].median())
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    for col in X_traditional.columns:
        if X_traditional[col].dtype == 'object':
            le = LabelEncoder()
            X_traditional[col] = le.fit_transform(X_traditional[col].astype(str))
    
    for col in X_behavioral.columns:
        if X_behavioral[col].dtype == 'object':
            le = LabelEncoder()
            X_behavioral[col] = le.fit_transform(X_behavioral[col].astype(str))
    
    # Get predictions from both models
    try:
        pred_traditional = model_traditional.predict_proba(X_traditional)[:, 1]
        print("✓ Generated predictions from traditional model")
    except Exception as e:
        print(f"⚠ Traditional model prediction failed: {e}")
        pred_traditional = np.zeros(len(df))
    
    try:
        pred_behavioral = model_behavioral.predict_proba(X_behavioral)[:, 1]
        print("✓ Generated predictions from behavioral model")
    except Exception as e:
        print(f"⚠ Behavioral model prediction failed: {e}")
        pred_behavioral = np.zeros(len(df))
    
    # Create meta-feature dataframe
    meta_features = pd.DataFrame({
        'pred_traditional': pred_traditional,
        'pred_behavioral': pred_behavioral,
        'pred_avg': (pred_traditional + pred_behavioral) / 2,
        'pred_max': np.maximum(pred_traditional, pred_behavioral),
        'pred_min': np.minimum(pred_traditional, pred_behavioral),
        'pred_diff': np.abs(pred_traditional - pred_behavioral),
        'pred_ratio': pred_traditional / (pred_behavioral + 0.001)
    })
    
    # Add some key features from both models (excluding ID columns)
    exclude_cols = ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU', 'ID', 'index']
    
    # Filter out ID columns before selecting key features
    filtered_traditional = [f for f in traditional_features if f not in exclude_cols]
    filtered_behavioral = [f for f in behavioral_features if f not in exclude_cols]
    
    key_traditional = filtered_traditional[:10] if len(filtered_traditional) >= 10 else filtered_traditional
    key_behavioral = filtered_behavioral[:10] if len(filtered_behavioral) >= 10 else filtered_behavioral
    
    for feat in key_traditional:
        if feat in X_traditional.columns:
            meta_features[f'trad_{feat}'] = X_traditional[feat].values
    
    for feat in key_behavioral:
        if feat in X_behavioral.columns:
            meta_features[f'behav_{feat}'] = X_behavioral[feat].values
    
    print(f"✓ Created {meta_features.shape[1]} meta-features")
    print(f"✓ Using {len(key_traditional)} key traditional features (excluding IDs)")
    print(f"✓ Using {len(key_behavioral)} key behavioral features (excluding IDs)")
    
    return meta_features

def train_ensemble_model(X_meta, y):
    """
    Train a meta-learner (LightGBM) on meta-features
    """
    print("\n" + "="*60)
    print("Training Ensemble Meta-Model")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train LightGBM meta-model
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*60}")
    print(f"Ensemble Model Performance")
    print(f"{'='*60}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, auc

def save_ensemble_model(model, model_traditional, model_behavioral, 
                       traditional_features, behavioral_features):
    """
    Save the ensemble model and metadata
    """
    models_dir = Path(__file__).parent.parent / "models"
    
    # Save ensemble model
    ensemble_path = models_dir / "model_ensemble_hybrid.pkl"
    joblib.dump(model, ensemble_path)
    print(f"\n✓ Saved ensemble model to {ensemble_path}")
    
    # Save metadata (feature lists and base models info)
    metadata = {
        'traditional_features': traditional_features,
        'behavioral_features': behavioral_features,
        'traditional_model_path': 'models/Traditional_model.pkl',
        'behavioral_model_path': 'models/Behavioral_model.pkl',
        'ensemble_type': 'stacking',
        'meta_learner': 'LightGBM'
    }
    
    metadata_path = models_dir / "ensemble_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"✓ Saved metadata to {metadata_path}")
    
    # Create and save wrapper
    wrapper = EnsembleHybridModel(
        model, model_traditional, model_behavioral,
        traditional_features, behavioral_features
    )
    
    wrapper_path = models_dir / "model_ensemble_wrapper.pkl"
    joblib.dump(wrapper, wrapper_path)
    print(f"✓ Saved ensemble wrapper to {wrapper_path}")
    
    return wrapper

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ENSEMBLE HYBRID MODEL TRAINING")
    print("="*70)
    
    # Load models
    model_traditional, model_behavioral = load_models()
    
    # Load hybrid data
    df = load_hybrid_data()
    
    # Check for TARGET
    if 'TARGET' not in df.columns:
        print("\n⚠ ERROR: No TARGET column found in hybrid dataset!")
        print("Please ensure smoke_hybrid_features.csv includes the TARGET column.")
        return
    
    # Drop rows with missing TARGET
    df_with_target = df[df['TARGET'].notna()].copy()
    print(f"\nRows with TARGET: {len(df_with_target)} / {len(df)}")
    
    if len(df_with_target) == 0:
        print("\n⚠ ERROR: No rows with valid TARGET values!")
        return
    
    df = df_with_target
    
    # Prepare feature sets
    traditional_features, behavioral_features = prepare_feature_sets(
        df, model_traditional, model_behavioral
    )
    
    # Create meta-features
    X_meta = create_meta_features(
        df, model_traditional, model_behavioral,
        traditional_features, behavioral_features
    )
    
    y = df['TARGET'].values
    
    # Train ensemble
    ensemble_model, auc = train_ensemble_model(X_meta, y)
    
    # Save models
    wrapper = save_ensemble_model(
        ensemble_model, model_traditional, model_behavioral,
        traditional_features, behavioral_features
    )
    
    print("\n" + "="*70)
    print("ENSEMBLE TRAINING COMPLETED!")
    print("="*70)
    print(f"\nFinal AUC-ROC: {auc:.4f}")
    print("\nSaved models:")
    print("  • model_ensemble_hybrid.pkl - Meta-learner model")
    print("  • model_ensemble_wrapper.pkl - Ready-to-use wrapper")
    print("  • ensemble_metadata.pkl - Feature and model metadata")

if __name__ == "__main__":
    main()
