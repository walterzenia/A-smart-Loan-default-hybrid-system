import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import joblib

# Load ensemble model
model = joblib.load('models/model_ensemble_wrapper.pkl')

print("Model type:", type(model))
print("Has meta_model:", hasattr(model, 'meta_model'))

meta = model.meta_model
print("\nMeta-model type:", type(meta))
print("Has feature_names_:", hasattr(meta, 'feature_names_'))
print("Has num_features_:", hasattr(meta, 'num_features_'))

if hasattr(meta, 'feature_names_'):
    print("\nFeature names:")
    for i, name in enumerate(meta.feature_names_):
        print(f"  {i}: {name}")
    print(f"\nTotal features: {len(meta.feature_names_)}")
else:
    print("\nNo feature_names_ attribute")
    
# Get feature importance
importance = meta.get_feature_importance()
print(f"\nFeature importance shape: {len(importance)}")
