"""
Fix ensemble wrapper to use only 7 meta-features (no additional key features)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import joblib
from ensemble_model import EnsembleHybridModel

print("="*70)
print("FIXING ENSEMBLE WRAPPER")
print("="*70)

# Load the CatBoost meta-model
print("\n1. Loading CatBoost meta-model...")
meta_model = joblib.load('models/model_ensemble_catboost_meta.pkl')
print(f"   ✓ Loaded meta-model with {len(meta_model.feature_names_)} features")
print(f"   Features: {list(meta_model.feature_names_)}")

# Load base models
print("\n2. Loading base models...")
model_trad = joblib.load('models/Traditional_model.pkl')
model_behav = joblib.load('models/Behaviorial_model.pkl')
print(f"   ✓ Traditional model: {len(model_trad.feature_name_)} features")
print(f"   ✓ Behavioral model: {len(model_behav.feature_name_)} features")

# Get feature lists
trad_features = model_trad.feature_name_
behav_features = model_behav.feature_name_

# Create new wrapper with corrected __init__ (no key features)
print("\n3. Creating new ensemble wrapper...")
ensemble_wrapper = EnsembleHybridModel(
    meta_model=meta_model,
    model_trad=model_trad,
    model_behav=model_behav,
    trad_feats=trad_features,
    behav_feats=behav_features
)

print(f"   ✓ Key traditional features: {len(ensemble_wrapper.key_traditional)}")
print(f"   ✓ Key behavioral features: {len(ensemble_wrapper.key_behavioral)}")

# Save corrected wrapper
print("\n4. Saving corrected wrapper...")
joblib.dump(ensemble_wrapper, 'models/model_ensemble_wrapper.pkl')
print("   ✓ Saved to models/model_ensemble_wrapper.pkl")

print("\n" + "="*70)
print("ENSEMBLE WRAPPER FIXED SUCCESSFULLY")
print("="*70)
print("\nThe model now uses only 7 meta-features:")
print("  1. pred_traditional")
print("  2. pred_behavioral")
print("  3. pred_avg")
print("  4. pred_max")
print("  5. pred_min")
print("  6. pred_diff")
print("  7. pred_ratio")
