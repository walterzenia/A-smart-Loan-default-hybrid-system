"""
Save test data for Traditional model using smoke_engineered.csv

Since the full traditional feature engineering requires too much memory,
we'll use the existing smoke_engineered.csv and split it the same way
as the original training would have done.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    print("Loading traditional_test_data.csv...")
    df = pd.read_csv("data/traditional_test_data.csv")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Check for target
    if 'TARGET' not in df.columns:
        print("ERROR: TARGET column not found!")
        return
    
    # Separate features and target
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Remove rows with NaN in target
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"After removing NaN targets: {len(y)} samples")
    print(f"Target distribution: {dict(y.value_counts())}")
    
    # Split with same parameters as training (test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Combine test features and target
    test_df = X_test.copy()
    test_df['TARGET'] = y_test
    
    # Save test data
    output_path = "models/Traditional_model_test_data.csv"
    test_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Test data saved to {output_path}")
    print(f"  - {len(test_df)} samples")
    print(f"  - {len(X_test.columns)} features + target")
    print(f"  - Target distribution: {dict(y_test.value_counts())}")

if __name__ == "__main__":
    main()
