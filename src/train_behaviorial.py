"""
Behavioral Model Training Script

Trains a LightGBM classifier using behavioral features from UCI Credit Card dataset.
This script uses pre-engineered features and does not duplicate feature engineering logic.

Usage:
    python train_behaviourial.py --data data/uci_credit_card.csv --output Behaviorial_model.pkl
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Import feature extraction function
from src.extract_features import behavioral_features


def parse_args():
    """Parse command line arguments for training."""
    p = argparse.ArgumentParser(description="Train behavioral model from UCI Credit Card data.")
    p.add_argument("--data", "-d", required=True, help="Path to UCI Credit Card CSV file")
    p.add_argument("--target", "-t", default="default.payment.next.month", help="Name of target column")
    p.add_argument(
        "--output", "-o", default="Behaviorial_model.pkl", help="Path to save trained model"
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


def train_model(X_train, y_train, X_valid=None, y_valid=None, random_state=42):
    """
    Train a LightGBM classifier for behavioral features.
    
    Features are already engineered by extract_features.py functions,
    so we just train the model directly without additional preprocessing.
    """
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=11,
        num_leaves=58,
        colsample_bytree=0.613,
        subsample=0.708,
        max_bin=407,
        reg_alpha=3.564,
        reg_lambda=4.930,
        min_child_weight=6,
        min_child_samples=165,
        random_state=random_state,
        n_jobs=4,
        verbose=-1
    )
    
    # Fit with validation set if provided
    if X_valid is not None and y_valid is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric='auc',
            eval_names=['training', 'valid_1']
        )
    else:
        model.fit(X_train, y_train)
    
    return model


def main():
    """Main training function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("train_behavioral")

    args = parse_args()

    # Load UCI Credit Card dataset
    logger.info("Loading UCI Credit Card dataset from %s...", args.data)
    uci_data = pd.read_csv(args.data)
    logger.info("Loaded dataset with shape: %s", uci_data.shape)

    # Extract behavioral features
    logger.info("Extracting behavioral features from UCI dataset...")
    features_df = behavioral_features(uci_data)
    logger.info("Behavioral features extracted: ~31 features")

    # Ensure target exists
    if args.target not in features_df.columns:
        logger.error("Target column '%s' not found in features", args.target)
        raise SystemExit(f"Target column '{args.target}' not found")

    # Use centralized data cleaning
    from src.data_cleaning import drop_id_columns, clean_target_column
    
    df = features_df
    df = drop_id_columns(df)
    df = clean_target_column(df, args.target)
    
    feature_cols = [c for c in df.columns if c != args.target]
    
    if not feature_cols:
        raise SystemExit("No feature columns found.")

    X = df[feature_cols]
    y = df[args.target]

    logger.info("Dataset shape: %s, Target distribution: %s", X.shape, dict(y.value_counts()))

    # Train/validation/test split
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Then split remaining into train and validation (80/20 split of remaining data)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=args.random_state, stratify=y_temp
    )
    
    logger.info("Train set: %d samples", len(X_train))
    logger.info("Validation set: %d samples", len(X_valid))
    logger.info("Test set: %d samples", len(X_test))

    logger.info("Training LightGBM model with validation...")
    model = train_model(X_train, y_train, X_valid, y_valid, random_state=args.random_state)

    # Evaluate
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, args.output)
    logger.info("Model saved to %s", args.output)
    print(f"\n✓ Model saved to {args.output}")
    
    # Save test set for validation and ROC curve generation
    test_data_path = args.output.replace('.pkl', '_test_data.csv')
    test_df = X_test.copy()
    test_df[args.target] = y_test
    test_df.to_csv(test_data_path, index=False)
    logger.info("Test data saved to %s", test_data_path)
    print(f"✓ Test data saved to {test_data_path}")
    print(f"  - {len(test_df)} samples")
    print(f"  - {len(feature_cols)} features + target")


if __name__ == "__main__":
    main()