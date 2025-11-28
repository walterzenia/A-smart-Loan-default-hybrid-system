"""
Traditional Model Training Script

Trains a LightGBM classifier using traditional features (all datasets combined).
This script uses pre-engineered features and does not duplicate feature engineering logic.

Usage:
    python train_traditional.py --output Traditional_model.pkl
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

# Import feature extraction functions (already handle all engineering)
from src.extract_features import traditional_features
from src.data_preprocessing import get_dataset


def parse_args():
    """Parse command line arguments for training."""
    p = argparse.ArgumentParser(description="Train traditional model from all datasets.")
    p.add_argument("--target", "-t", default="TARGET", help="Name of target column")
    p.add_argument(
        "--output", "-o", default="Traditional_model.pkl", help="Path to save trained model"
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


def train_model(X_train, y_train, random_state=42):
    """
    Train a LightGBM classifier.
    
    Features are already engineered by extract_features.py functions,
    so we just train the model directly without additional preprocessing.
    """
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    return model


def main():
    """Main training function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("train_traditional")

    args = parse_args()

    logger.info("Loading datasets from data/ directory...")
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()

    # Extract traditional features (all datasets combined)
    logger.info("Extracting traditional features from all datasets...")
    features_df = traditional_features(apps, bureau, bureau_bal, prev, pos_bal, install, card_bal)
    logger.info("Traditional features extracted: ~487 features")

    # Ensure target exists and drop rows without target
    if args.target not in features_df.columns:
        logger.error("Target column '%s' not found in features", args.target)
        raise SystemExit(f"Target column '{args.target}' not found")

    df = features_df.dropna(subset=[args.target])
    feature_cols = [c for c in df.columns if c != args.target]
    
    if not feature_cols:
        raise SystemExit("No feature columns found.")

    X = df[feature_cols]
    y = df[args.target]

    logger.info("Dataset shape: %s, Target distribution: %s", X.shape, dict(y.value_counts()))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    logger.info("Training LightGBM model...")
    model = train_model(X_train, y_train, random_state=args.random_state)

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


if __name__ == "__main__":
    main()